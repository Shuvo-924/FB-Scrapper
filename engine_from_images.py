"""
Engine number from post photos AND reel/video frames, for journeys where no post text stated it.
  YOLO plate detector  ->  Geometric Gatekeeper (No words/stickers)  ->  Best Crops  ->  Gemini Vision Reader
  -> Vote across crops, posts, and journey  ->  output/engine_from_images.csv
"""
import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import shutil
import time
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import Callable, Iterator, List, Optional, Tuple
import cv2
import numpy as np
import pandas as pd
from google.genai import types
from pydantic import BaseModel
from build_dataset import (BN_TO_EN, CACHE_FILE, DIR_TO_TRAIN, LOCO_MAX, LOCO_MIN, MODELS, OUT, TRAIN_ALIASES,
                           VALID_EVENTS, _nm, assign_train, canonical_station, client, norm_train, parse_clock,
                           parse_post_dt, post_key, prep_text, resolve_actual_dt)
IMG_SCHEMA = "img-v4-ytdlp-ffmpeg-30fps"
YOLO_WEIGHTS = Path(os.getenv("YOLO_WEIGHTS", "runs/detect/loco_plate_detector/weights/best.pt"))
DIGIT_WEIGHTS = Path(os.getenv("DIGIT_WEIGHTS", "runs/detect/loco_digit_detector/weights/best.pt"))
VISION_MODELS = [m.strip() for m in os.getenv("VISION_MODELS", ",".join(MODELS)).split(",") if m.strip()]
VISION_PAUSE_S = float(os.getenv("VISION_PAUSE_S", "4.2"))
# --- TUNED DETECTION & GEOMETRIC GATEKEEPER ---
YOLO_CONF = float(os.getenv("YOLO_CONF", "0.45"))                    # Increased from 0.25 to eliminate false text boxes
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))
MIN_PLATE_W = int(os.getenv("MIN_PLATE_W", "45"))                   # 45px minimum (below this, digits are unreadable)
MIN_ASPECT = 1.35                                                   # Drops square stickers like "ছেড়ে"
MAX_ASPECT = 4.20                                                   # Drops wide coach nameboards like "কালনী এক্সপ্রেস"
PLATE_PAD = 0.10
TOP_CROPS = int(os.getenv("TOP_CROPS", "6"))
MIN_GAP_S = float(os.getenv("MIN_GAP_S", "0.4"))
KEEP_RATIO = float(os.getenv("KEEP_RATIO", "0.5"))
MIN_CROP_QUALITY = float(os.getenv("MIN_CROP_QUALITY", "0.15"))
VIDEO_FPS = float(os.getenv("VIDEO_FPS", "30"))
VIDEO_MAX_SEC = float(os.getenv("VIDEO_MAX_SEC", "40"))  # set 0 to process the full video
MAX_POST_PHOTOS = int(os.getenv("ENGINE_MAX_POST_PHOTOS", "10"))
YTDLP_BIN = os.getenv("YTDLP_BIN", "yt-dlp")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
YTDLP_TIMEOUT = int(os.getenv("YTDLP_TIMEOUT", "600"))
INCLUDE_VIDEO = os.getenv("ENGINE_INCLUDE_VIDEO", "1") == "1"
MIN_READ_WEIGHT = 0.35
MIN_VOTES = int(os.getenv("ENGINE_MIN_VOTES", "2"))
MIN_POSTS = int(os.getenv("ENGINE_MIN_POSTS", "2"))
ACCEPT_SCORE = float(os.getenv("ENGINE_ACCEPT_SCORE", "1.4"))
SINGLE_CROP_SCORE = float(os.getenv("ENGINE_SINGLE_SCORE", "1.05"))
UNKNOWN_PENALTY = 0.75
WINDOW_PAD_MIN = 60
IMG_CACHE = OUT / "engine_image_cache.json"
MEDIA_CACHE_FILE = OUT / "engine_media_cache.json"
CROP_DIR = OUT / "engine_crops"
CROP_DIR.mkdir(exist_ok=True)
VIDEO_DIR = OUT / "engine_videos"
VIDEO_DIR.mkdir(exist_ok=True)
PHOTO_DIR = OUT / "engine_photos"
PHOTO_DIR.mkdir(exist_ok=True)
# =========================================================================
# 1. VISION READER & SCHEMA
# =========================================================================
class CropRead(BaseModel):
    crop_index: int
    locomotive_number: Optional[str] = None     # exactly 4 digits, only if clearly legible
    confidence: float = 0.0
    digits_seen: Optional[str] = None
    note: Optional[str] = None
VISION_PROMPT = """
You read the 4-digit road number of Bangladesh Railway diesel locomotives from tight crops of number plates.
CRITICAL REJECTION RULES (DO NOT GUESS):
1. If the crop contains Bengali WORDS (e.g. 'ছেড়ে', 'কালনী', 'এক্সপ্রেস', station boards, or stickers), it is NOT a locomotive plate. Return locomotive_number = null!
2. A valid plate contains ONLY 4 digits (e.g. 2901, 2908, 2936, 3026 or ২৯০১, ২৯০৮). Numbers are between 2600 and 3399.
3. If blurry, dark, motion-smeared, or unreadable: return null, keep confidence low (< 0.5).
4. Do NOT confuse 3-digit train numbers (773, 774) or years (2026) with engine numbers.
5. Digits may be Bengali (০-৯): ALWAYS output English digits.
"""
def _digits(s) -> Optional[str]:
    d = re.sub(r"\D", "", str(s or "").translate(BN_TO_EN))
    return d if re.fullmatch(r"\d{4}", d) and LOCO_MIN <= int(d) <= LOCO_MAX else None
def gemini_read(post_text: str, jpgs: List[bytes]) -> Optional[List[dict]]:
    contents = [VISION_PROMPT + f"\nPOST TEXT (context only, never source of digits):\n{prep_text(post_text)[:250]}\n\nCROPS:"]
    for i, jpg in enumerate(jpgs):
        contents += [f"Crop {i}:", types.Part.from_bytes(data=jpg, mime_type="image/jpeg")]
    for model in VISION_MODELS:
        delay = 4
        for attempt in range(1, 4):
            try:
                resp = client.models.generate_content(
                    model=model, contents=contents,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[CropRead],
                        temperature=0.0
                    )
                )
                time.sleep(VISION_PAUSE_S)
                if resp.parsed:
                    return [r.model_dump() for r in resp.parsed]
                break
            except Exception as e:
                err = str(e)
                if any(c in err for c in ("503", "429", "UNAVAILABLE")):
                    time.sleep(delay)
                    delay = min(delay * 2, 30)
                else:
                    break
    return None
class PlateDetector:
    def __init__(self, weights: Path):
        from ultralytics import YOLO
        self.model = YOLO(str(weights))
    def __call__(self, bgr):
        out = []
        for r in self.model.predict(bgr, conf=YOLO_CONF, imgsz=YOLO_IMGSZ, verbose=False):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                out.append((*map(int, b), float(c)))
        return out
# =========================================================================
# 2. FRAMES -> GEOMETRIC FILTERING & BEST CROPS
# =========================================================================
@dataclass
class Crop:
    bgr: np.ndarray
    conf: float
    quality: float
    t: Optional[float]
    w: int
def crop_quality(crop, conf: float) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, max(16, int(160 * gray.shape[0] / max(1, gray.shape[1])))))
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharp = lap / (lap + 180.0)
    size = min(1.0, crop.shape[1] / 120.0)
    return float(conf * (0.2 + 0.8 * sharp) * (0.3 + 0.7 * size))
def collect_crops(frames: Iterator[Tuple[np.ndarray, Optional[float]]], detect: Callable) -> List[Crop]:
    out = []
    for bgr, t in frames:
        H, W = bgr.shape[:2]
        for x1, y1, x2, y2, conf in detect(bgr):
            # Geometric Gatekeeper: Discard non-plate proportions before saving
            w = x2 - x1
            h = y2 - y1
            if w < MIN_PLATE_W or h < 14:
                continue
            aspect = w / max(1, h)
            # Locomotive plates are strictly between 1.35 and 4.2 wide
            # Rejects square stickers ("ছেড়ে" ~ 1.1) and wide coach banners ("এক্সপ্রেস" > 4.5)
            if aspect < MIN_ASPECT or aspect > MAX_ASPECT:
                continue
            # Add tight padding
            px, py = int(w * PLATE_PAD), int(h * PLATE_PAD)
            x1, y1 = max(0, x1 - px), max(0, y1 - py)
            x2, y2 = min(W, x2 + px), min(H, y2 + py)
            crop = bgr[y1:y2, x1:x2].copy()
            out.append(Crop(crop, conf, crop_quality(crop, conf), t, x2 - x1))
    return out
def select_top(crops: List[Crop], k: int) -> List[Crop]:
    if not crops:
        return []
    floor = max(MIN_CROP_QUALITY, KEEP_RATIO * max(c.quality for c in crops))
    chosen = []
    for c in sorted((c for c in crops if c.quality >= floor), key=lambda c: -c.quality):
        if any(c.t is not None and o.t is not None and abs(c.t - o.t) < MIN_GAP_S for o in chosen):
            continue
        chosen.append(c)
        if len(chosen) == k:
            break
    return chosen
def upscale(crop, min_h: int = 160, max_scale: float = 4.0):
    h = crop.shape[0]
    if h >= min_h:
        return crop
    s = min(max_scale, min_h / h)
    return cv2.resize(crop, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
def prep_crop_jpg(crop) -> bytes:
    big = upscale(crop)
    blur = cv2.GaussianBlur(big, (0, 0), 1.0)
    sharp = cv2.addWeighted(big, 1.3, blur, -0.3, 0)
    ok, buf = cv2.imencode(".jpg", sharp, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return buf.tobytes()
def iter_video_frames(path: Path, fps: float = None, max_sec: float = None):
    """Extract frames with FFmpeg at the requested FPS, retaining original video resolution."""
    fps = float(VIDEO_FPS if fps is None else fps)
    max_sec = VIDEO_MAX_SEC if max_sec is None else float(max_sec)
    if fps <= 0:
        raise ValueError("VIDEO_FPS must be greater than 0")
    if shutil.which(FFMPEG_BIN) is None:
        raise RuntimeError(f"FFmpeg was not found: {FFMPEG_BIN!r}. Install ffmpeg and ensure it is on PATH.")

    fps_tag = str(fps).replace(".", "p")
    frame_dir = VIDEO_DIR / f"{path.stem}_frames_{fps_tag}fps"
    complete_marker = frame_dir / ".complete"
    frame_dir.mkdir(parents=True, exist_ok=True)
    frames = sorted(frame_dir.glob("frame_*.jpg"))
    expected_marker = f"fps={fps:g}\nmax_sec={max_sec:g}\n"
    marker_matches = complete_marker.exists() and complete_marker.read_text(encoding="utf-8", errors="ignore") == expected_marker
    if not marker_matches or not frames:
        for old in frame_dir.glob("frame_*.jpg"):
            old.unlink(missing_ok=True)
        complete_marker.unlink(missing_ok=True)
        pattern = str(frame_dir / "frame_%07d.jpg")
        cmd = [
            FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(path), "-vf", f"fps={fps:g}", "-q:v", "2",
        ]
        if max_sec and max_sec > 0:
            cmd += ["-t", str(max_sec)]
        cmd.append(pattern)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg frame extraction failed: {(proc.stderr or proc.stdout)[-1200:]}")
        frames = sorted(frame_dir.glob("frame_*.jpg"))
        if frames:
            complete_marker.write_text(expected_marker, encoding="utf-8")

    for index, frame_path in enumerate(frames):
        bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if bgr is not None:
            yield bgr, index / fps
# =========================================================================
# 3. MEDIA PROBE, FULL-RES PHOTOS, YT-DLP VIA CDP COOKIES, FFMPEG FRAMES
# =========================================================================
MEDIA_PROBE_JS = r"""
() => {
  const root = document.querySelector('div[role="dialog"]') ||
               document.querySelector('div[role="main"]') || document.body;
  const visible = el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width >= 100 && r.height >= 80 && s.display !== 'none' &&
           s.visibility !== 'hidden' && Number(s.opacity || 1) > 0;
  };
  const pathLooksVideo = /\/(reel|videos|watch|live)\//i.test(location.pathname);
  const videos = [...root.querySelectorAll('video')].filter(v => {
    const r = v.getBoundingClientRect();
    const source = v.currentSrc || v.src || v.poster || v.querySelector('source')?.src;
    return source && r.width >= 120 && r.height >= 80;
  });
  const images = [...root.querySelectorAll('img')].filter(im => {
    const r = im.getBoundingClientRect();
    return im.complete && im.naturalWidth >= 250 && r.width >= 160 && r.height >= 100 && visible(im);
  });
  const photoLinks = [...root.querySelectorAll('a[href]')].some(a =>
    /\/photo\/|photo\.php|\/photos\//i.test(a.href) && visible(a));
  return {
    hasVideo: pathLooksVideo || videos.length > 0,
    hasPhoto: images.length > 0 || photoLinks || /\/(photo|photos)\//i.test(location.pathname),
    imageCount: images.length,
    videoCount: videos.length
  };
}
"""

IMG_JS = r"""
() => {
  const root = document.querySelector('div[role="dialog"]') ||
               document.querySelector('div[role="main"]') || document.body;
  const out = [];
  let i = 0;
  const visible = el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    return r.width >= 160 && r.height >= 100 && s.display !== 'none' &&
           s.visibility !== 'hidden' && Number(s.opacity || 1) > 0;
  };
  for (const im of root.querySelectorAll('img')) {
    const r = im.getBoundingClientRect();
    if (!im.complete || im.naturalWidth < 250 || !visible(im)) continue;
    const choices = [];
    const add = (url, width) => {
      if (url && /^https?:/i.test(url)) choices.push({url, width: Number(width) || 0});
    };
    // Facebook commonly provides several transformed resolutions through srcset.
    for (const part of (im.srcset || '').split(',')) {
      const tokens = part.trim().split(/\s+/);
      if (tokens.length) {
        const descriptor = tokens[1] || '';
        const width = descriptor.endsWith('w') ? parseInt(descriptor, 10) :
                      descriptor.endsWith('x') ? parseFloat(descriptor) * im.naturalWidth : im.naturalWidth;
        add(tokens[0], width);
      }
    }
    add(im.currentSrc, im.naturalWidth);
    add(im.src, im.naturalWidth);
    const seen = new Set();
    const unique = choices.filter(c => !seen.has(c.url) && seen.add(c.url));
    unique.sort((a, b) => b.width - a.width);
    if (!unique.length) continue;
    im.setAttribute('data-eng-img', String(i));
    out.push({i: i++, src: unique[0].url, srcCandidates: unique.map(c => c.url),
              width: unique[0].width || im.naturalWidth,
              area: im.naturalWidth * im.naturalHeight, displayArea: r.width * r.height});
  }
  out.sort((a, b) => (b.width * Math.max(1, b.displayArea)) - (a.width * Math.max(1, a.displayArea)));
  return out;
}
"""


def _explicit_media_hint(post: dict) -> Optional[bool]:
    """Use explicit media fields if the scraper happened to save them; otherwise probe the page."""
    if isinstance(post.get("has_media"), bool):
        return post["has_media"]
    positive_keys = ("attachments", "media", "photos", "videos", "images", "photo_urls",
                     "video_urls", "media_urls", "image_url", "video_url", "thumbnail_url")
    for key in positive_keys:
        value = post.get(key)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, (list, tuple, dict)) and len(value) > 0:
            return True
    url = str(post.get("url") or "")
    if re.search(r"/(?:reel|videos|watch|photo|photos)/|photo\.php", url, re.I):
        return True
    return None


def post_has_media(context, post: dict) -> Optional[bool]:
    """Quick preflight: skip text-only posts before YOLO/Gemini is invoked."""
    explicit = _explicit_media_hint(post)
    if explicit is not None:
        return explicit
    url = str(post.get("url") or "")
    if not url:
        return False
    pg = context.new_page()
    try:
        pg.goto(url, wait_until="domcontentloaded", timeout=25000)
        pg.wait_for_timeout(1200)
        info = pg.evaluate(MEDIA_PROBE_JS)
        return bool(info.get("hasPhoto") or info.get("hasVideo"))
    except Exception as e:
        print(f"      [!] media preflight uncertain: {url[-35:]} | {str(e)[:100]}", flush=True)
        return None  # don't persist a transient failure as a definitive no-media result
    finally:
        try:
            pg.close()
        except Exception:
            pass


def _write_netscape_cookies(context, cookie_path: Path) -> int:
    """Export the already-authenticated CDP browser's cookies to a temporary yt-dlp cookie jar."""
    cookies = context.cookies()
    lines = ["# Netscape HTTP Cookie File", "# Temporary cookies exported from the active Playwright CDP context"]
    written = 0
    for c in cookies:
        domain = str(c.get("domain") or "")
        name = str(c.get("name") or "")
        value = str(c.get("value") or "")
        if not domain or not name:
            continue
        include_subdomains = "TRUE" if domain.startswith(".") else "FALSE"
        path = str(c.get("path") or "/")
        secure = "TRUE" if c.get("secure") else "FALSE"
        expires = c.get("expires", -1)
        try:
            expires = int(expires) if float(expires) > 0 else 0
        except (TypeError, ValueError):
            expires = 0
        if c.get("httpOnly"):
            domain = "#HttpOnly_" + domain
        # Cookie values normally don't contain tabs/newlines; skip malformed records defensively.
        if any(ch in value + name + domain + path for ch in "\r\n\t"):
            continue
        lines.append("\t".join([domain, include_subdomains, path, secure, str(expires), name, value]))
        written += 1
    cookie_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    try:
        os.chmod(cookie_path, 0o600)
    except OSError:
        pass
    return written


def download_reel_ytdlp(context, post_url: str) -> Optional[Path]:
    """Download the best available Reel/video formats using cookies from the current CDP session."""
    if shutil.which(YTDLP_BIN) is None:
        print(f"      [!] {YTDLP_BIN!r} not found. Install with: pip install -U yt-dlp", flush=True)
        return None
    key = hashlib.md5(post_url.encode("utf-8")).hexdigest()[:12]
    existing = [p for p in VIDEO_DIR.glob(f"{key}.*") if p.is_file() and p.suffix.lower() not in {".part", ".ytdl", ".json"}]
    if existing:
        return max(existing, key=lambda p: p.stat().st_size)

    fd, cookie_name = tempfile.mkstemp(prefix="yt_dlp_cdp_", suffix=".cookies.txt", dir=str(VIDEO_DIR))
    os.close(fd)
    cookie_path = Path(cookie_name)
    try:
        cookie_count = _write_netscape_cookies(context, cookie_path)
        if cookie_count == 0:
            print("      [!] CDP context has no cookies; yt-dlp may not be authenticated.", flush=True)
        output_template = str(VIDEO_DIR / f"{key}.%(ext)s")
        cmd = [
            YTDLP_BIN,
            "--cookies", str(cookie_path),
            "--no-playlist", "--no-progress", "--no-warnings",
            "--format", "bestvideo*+bestaudio/best",
            "--merge-output-format", "mkv",
            "--output", output_template,
            "--print", "after_move:filepath",
            post_url,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=YTDLP_TIMEOUT)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "yt-dlp failed without a message").strip()
            print(f"      [!] yt-dlp failed for {post_url[-35:]}: {detail[-1200:]}", flush=True)
            return None
        # yt-dlp's after_move print is the most reliable output path; fall back to the hashed files.
        for line in reversed((proc.stdout or "").splitlines()):
            candidate = Path(line.strip().strip('"'))
            if candidate.is_file():
                return candidate
        files = [p for p in VIDEO_DIR.glob(f"{key}.*") if p.is_file() and p.suffix.lower() not in {".part", ".ytdl", ".json"}]
        return max(files, key=lambda p: p.stat().st_size) if files else None
    except subprocess.TimeoutExpired:
        print(f"      [!] yt-dlp timed out after {YTDLP_TIMEOUT}s: {post_url[-35:]}", flush=True)
        return None
    except Exception as e:
        print(f"      [!] yt-dlp error: {str(e)[:200]}", flush=True)
        return None
    finally:
        cookie_path.unlink(missing_ok=True)


def _download_best_photo(context, pg, info: dict, post_url: str) -> Optional[np.ndarray]:
    """Download and save the largest DOM-exposed image candidate; screenshot is fallback only."""
    candidates = info.get("srcCandidates") or [info.get("src")]
    data = None
    media_ext = ".jpg"
    for image_url in candidates:
        if not image_url:
            continue
        try:
            response = context.request.get(image_url, timeout=15000)
            if response.ok:
                body = response.body()
                if len(body) > 8000:
                    data = body
                    content_type = (response.headers.get("content-type") or "").split(";", 1)[0].lower()
                    media_ext = {"image/png": ".png", "image/webp": ".webp", "image/gif": ".gif",
                                 "image/jpeg": ".jpg", "image/avif": ".avif"}.get(content_type, ".jpg")
                    break
        except Exception:
            continue
    if not data:
        try:
            data = pg.locator(f'[data-eng-img="{info["i"]}"]').first.screenshot(
                type="jpeg", quality=98, timeout=5000
            )
            media_ext = ".jpg"
        except Exception:
            return None
    # Preserve the original downloaded attachment for later audits/re-annotation.
    photo_key = hashlib.md5(
        f"{post_url}|{info.get('src')}|{info.get('i')}".encode("utf-8")
    ).hexdigest()[:16]
    photo_path = PHOTO_DIR / f"{photo_key}{media_ext}"
    if not photo_path.exists():
        photo_path.write_bytes(data)
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)


def browser_frames(context, url: str) -> Iterator[Tuple[np.ndarray, Optional[float]]]:
    """Yield highest-resolution attached photos and 30-fps FFmpeg frames for video/Reel posts."""
    pg = context.new_page()
    try:
        pg.goto(url, wait_until="domcontentloaded", timeout=30000)
        pg.wait_for_timeout(1400)
        media = pg.evaluate(MEDIA_PROBE_JS)

        # Photos: choose the largest width advertised by Facebook's image srcset, not a screenshot first.
        images = pg.evaluate(IMG_JS)
        n_images = 0
        for info in images[:MAX_POST_PHOTOS]:
            bgr = _download_best_photo(context, pg, info, url)
            if bgr is not None and bgr.size:
                n_images += 1
                yield bgr, None

        # A post may contain both photos and video, so video processing is independent of photo count.
        path_is_video = bool(re.search(r"/(?:reel|videos|watch|live)/", url, re.I))
        if INCLUDE_VIDEO and (media.get("hasVideo") or path_is_video):
            video_path = download_reel_ytdlp(context, url)
            if video_path and video_path.exists():
                yield from iter_video_frames(video_path, fps=VIDEO_FPS, max_sec=VIDEO_MAX_SEC)
            else:
                print(f"      [!] no downloaded video; could not process Reel: {url[-40:]}", flush=True)

        if not n_images and not (INCLUDE_VIDEO and (media.get("hasVideo") or path_is_video)):
            return
    except Exception as e:
        print(f"      [!] media error: {str(e)[:160]}", flush=True)
    finally:
        try:
            pg.close()
        except Exception:
            pass


# =========================================================================
# 4. SCORING & PIPELINE EXECUTION
# =========================================================================
_OTHER_NAMES = [_nm(a.translate(BN_TO_EN)) for al in TRAIN_ALIASES.values() for a in al if not re.fullmatch(r"[\d০-৯]+", a)]
def post_weight(text: str) -> float:
    t = _nm(str(text or "").translate(BN_TO_EN))
    kalni = any(k in t for k in ("কালনী", "kalni", "773", "774"))
    other = any(n in t for n in _OTHER_NAMES)
    return 0.0 if (other and not kalni) else 0.5 if other else 1.0
def read_post(post, frames_fn, detect, gemini_fn, prefix: str) -> dict:
    text, url = post.get("text", ""), post["url"]
    w = post_weight(text)
    if w == 0:
        return {"status": "other_train", "url": url, "reads": []}
    n_frames = 0
    def counted():
        nonlocal n_frames
        for f in frames_fn(url):
            n_frames += 1
            yield f
    crops = collect_crops(counted(), detect)
    if n_frames == 0:
        return {"status": "no_media", "url": url, "reads": []}
    top = select_top(crops, TOP_CROPS)
    if not top:
        return {"status": "no_plate", "url": url, "reads": [], "n_frames": n_frames}
    jpgs, files = [], []
    for i, c in enumerate(top):
        jpg = prep_crop_jpg(c.bgr)
        f = CROP_DIR / f"{prefix}_c{i}.jpg"
        f.write_bytes(jpg)
        jpgs.append(jpg)
        files.append(f.name)
    g_out = gemini_fn(text, jpgs)
    if g_out is None:
        return {"status": "model_failed", "url": url, "reads": []}
    reads = []
    for i, c in enumerate(top):
        gi = next((x for x in (g_out or []) if x.get("crop_index") == i), None)
        num = _digits(gi.get("locomotive_number")) if gi else None
        conf = float(gi.get("confidence") or 0) if gi else 0.0
        if num and conf >= 0.5:
            reads.append({
                "crop_id": f"{prefix}#{i}",
                "number": num,
                "weight": conf * w,
                "t": c.t,
                "crop_file": files[i]
            })
    return {"status": "read", "url": url, "reads": reads, "n_frames": n_frames, "n_crops": len(top)}
def journey_window(r):
    dep = datetime.strptime(f"{r['journey_date']} {r.get('departure_time') or r['scheduled_departure']}", "%Y-%m-%d %H:%M")
    if r["train_number"] == "773" and dep.hour < 6:
        dep += timedelta(days=1)
    arr = (datetime.strptime(f"{r['destination_arrival_date']} {r['destination_arrival_time']}", "%Y-%m-%d %H:%M")
           if r.get("destination_arrival_time") and r.get("destination_arrival_date") else dep + timedelta(hours=8))
    pad = timedelta(minutes=WINDOW_PAD_MIN)
    return dep - pad, arr + pad, dep
def assign_posts(posts, text_cache, rows):
    windows = {(r["train_number"], r["journey_date"]): journey_window(r) for r in rows}
    by_j, skipped = defaultdict(list), 0
    for p in posts:
        ex, post_dt = text_cache.get(post_key(p)), parse_post_dt(p.get("date_time"))
        if not post_dt or not p.get("url"):
            continue
        hint_t = norm_train((ex or {}).get("train_number"))
        hint_d = ((ex or {}).get("direction") or "").strip().lower()
        keys = []
        for o in (ex or {}).get("observations") or []:
            event, st, clock = (o.get("event") or "").lower(), canonical_station(o.get("station")), parse_clock(o.get("time_text"))
            if event in VALID_EVENTS and st and clock:
                res = assign_train(st, event, resolve_actual_dt(post_dt, *clock), hint_t, hint_d)
                if res:
                    keys.append((res[0], res[1].isoformat()))
        key = Counter(keys).most_common(1)[0][0] if keys else None
        if key is None:
            want = hint_t or DIR_TO_TRAIN.get(hint_d)
            cands = [k for k, (a, b, _) in windows.items() if a <= post_dt <= b and (not want or k[0] == want)]
            key = cands[0] if len(cands) == 1 else None
        if key is None:
            skipped += 1
            continue
        by_j[key].append((post_dt, p))
    return by_j
def decide(reads, known):
    score, crops, posts, best_one = Counter(), defaultdict(set), defaultdict(set), Counter()
    for r in reads:
        if r["weight"] < MIN_READ_WEIGHT:
            continue
        w = r["weight"] * (1.0 if (not known or r["number"] in known) else UNKNOWN_PENALTY)
        score[r["number"]] += w
        best_one[r["number"]] = max(best_one[r["number"]], w)
        crops[r["number"]].add(r["crop_id"])
        posts[r["number"]].add(r["crop_id"].split("#")[0])
    if not score:
        return None
    ranked = score.most_common()
    top, s1 = ranked[0]
    s2 = ranked[1][1] if len(ranked) > 1 else 0.0
    clear = s1 >= 1.4 * s2
    multi = len(crops[top]) >= MIN_VOTES and s1 >= ACCEPT_SCORE
    single = best_one[top] >= SINGLE_CROP_SCORE and (not known or top in known)
    return {"number": top, "score": round(s1, 2), "votes": len(crops[top]), "posts": len(posts[top]),
            "accepted": clear and (multi or single)}
def run(rows, posts, text_cache, frames_fn, detect, gemini_fn, has_media_fn=None, validate=False, limit=None):
    from_text = lambda r: bool(r.get("loco_primary")) and r.get("loco_source", "text") in ("text", "")
    known = {r["loco_primary"] for r in rows if from_text(r)}
    by_j = assign_posts(posts, text_cache, rows)
    cache = json.loads(IMG_CACHE.read_text(encoding="utf-8")) if IMG_CACHE.exists() else {}
    media_cache = json.loads(MEDIA_CACHE_FILE.read_text(encoding="utf-8")) if MEDIA_CACHE_FILE.exists() else {}
    targets = [r for r in rows if from_text(r) == validate]
    if validate:
        targets = targets[:limit or 30]
    print(f"[+] Journeys to inspect: {len(targets)} ({'Validation Mode' if validate else 'Missing Engine Fill'})", flush=True)
    results = []
    for r in targets:
        key = (r["train_number"], r["journey_date"])
        assigned_posts = sorted(by_j.get(key, []), key=lambda x: x[0])
        media_cands = []
        for post_dt, post in assigned_posts:
            post_url = str(post.get("url") or "")
            if not post_url:
                continue
            media_flag = media_cache.get(post_url)
            if media_flag is None:
                if has_media_fn is None:
                    # Without a probe callback, defer media detection to the frame reader.
                    media_flag = True
                else:
                    media_flag = has_media_fn(post)
                if media_flag is not None:
                    media_cache[post_url] = bool(media_flag)
                    MEDIA_CACHE_FILE.write_text(json.dumps(media_cache, ensure_ascii=False, indent=2), encoding="utf-8")
            if media_flag is not False:
                # Include an uncertain probe result for one actual processing attempt; only confirmed no-media is skipped.
                media_cands.append((post_dt, post))
            else:
                print(f"      [skip no-media] {post_url[-45:]}", flush=True)
        cands = sorted(
            media_cands,
            key=lambda x: (0 if re.search(r"/(?:photo|photos)/|photo\.php", x[1]["url"], re.I) else 1, x[0])
        )[:5]
        reads, dec = [], None
        for post_dt, p in cands:
            ck = hashlib.md5(f"{IMG_SCHEMA}|{p['url']}".encode()).hexdigest()
            res = cache.get(ck)
            if res is None:
                res = read_post(p, frames_fn, detect, gemini_fn, prefix=ck[:10])
                if res["status"] != "model_failed":
                    cache[ck] = res
                    IMG_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            for rd in res["reads"]:
                reads.append(dict(rd, post_dt=post_dt, url=p["url"]))
            top = Counter(x["number"] for x in res["reads"]).most_common(1)
            print(f"    {key[1]} {key[0]} | {p['url'][-30:]:30s} {res['status']:11s} {top[0][0] if top else '-'}", flush=True)
            dec = decide(reads, known)
            if dec and dec["accepted"]:
                break
        row = {
            "journey_date": key[1], "train_number": key[0], "n_assigned_posts": len(assigned_posts),
            "n_skipped_no_media": sum(1 for _, p in assigned_posts if media_cache.get(str(p.get("url") or "")) is False),
            "n_candidate_posts": len(cands), "n_reads": len(reads),
            "loco_from_image": dec["number"] if dec and dec["accepted"] else None,
            "loco_image_score": dec["score"] if dec else None, "loco_image_votes": dec["votes"] if dec else 0,
            "loco_image_posts": dec["posts"] if dec else 0,
            "best_number": dec["number"] if dec else None
        }
        if dec and dec["accepted"]:
            mine = [x for x in reads if x["number"] == dec["number"]]
            first = min(mine, key=lambda x: x["post_dt"])
            row["loco_image_pre_departure"] = int(first["post_dt"] <= journey_window(r)[2] + timedelta(minutes=10))
        results.append(row)
    return pd.DataFrame(results)
def is_port_listening(port=9222, host="127.0.0.1"):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False
def main():
    from playwright.sync_api import sync_playwright
    if not YOLO_WEIGHTS.exists():
        raise SystemExit(f"[-] YOLO weights not found at {YOLO_WEIGHTS}")
    detect = PlateDetector(YOLO_WEIGHTS)
    print(f"[+] Loaded YOLO detector: {YOLO_WEIGHTS} (Conf: {YOLO_CONF}, Aspect: {MIN_ASPECT}-{MAX_ASPECT})", flush=True)
    raw = json.loads((OUT / "posts.json").read_text(encoding="utf-8"))
    seen, posts = set(), []
    for p in raw:
        k = post_key(p)
        dedupe_key = (k, p.get("url", ""))
        if p.get("url") and dedupe_key not in seen:
            seen.add(dedupe_key)
            posts.append(p)
    text_cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    rows = pd.read_csv(OUT / "kalni_journeys_all.csv", dtype=str).fillna("").to_dict("records")
    with sync_playwright() as p:
        if not is_port_listening(9222):
            raise SystemExit("[-] Start Brave first: /snap/bin/brave --remote-debugging-port=9222 > /dev/null 2>&1 &")
        context = p.chromium.connect_over_cdp("http://127.0.0.1:9222").contexts[0]
        df = run(rows, posts, text_cache, lambda u: browser_frames(context, u), detect, gemini_read,
                  has_media_fn=lambda post: post_has_media(context, post))
    df.to_csv(OUT / "engine_from_images.csv", index=False, encoding="utf-8-sig")
    print(f"\n[DONE] Verified engines saved to: {OUT / 'engine_from_images.csv'}")
    
if __name__ == "__main__":
    main()