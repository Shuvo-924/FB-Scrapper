import io
import json
import os
import re
import socket
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image
from playwright.sync_api import sync_playwright
from pydantic import BaseModel, Field
from ultralytics import YOLO

# =========================================================================
# SETUP & PATHS
# =========================================================================
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY! Set it in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)
VISION_MODEL = "gemini-3.5-flash-lite"  # Free tier: 500 requests/day, 15 RPM

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

OUT_DIR = Path("output")
IMAGES_DIR = OUT_DIR / "post_images"
CROPS_DIR = OUT_DIR / "plate_crops"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CROPS_DIR.mkdir(parents=True, exist_ok=True)

ENGINE_CSV_FILE = OUT_DIR / "engine_from_images.csv"
POSTS_JSON_FILE = OUT_DIR / "posts.json"
JOURNEYS_CSV_FILE = OUT_DIR / "kalni_journeys_all.csv"

# Path to your trained YOLO plate detector weights
YOLO_WEIGHTS_PATH = Path("runs/detect/loco_plate_detector/weights/best.pt")
yolo_detector = YOLO(str(YOLO_WEIGHTS_PATH)) if YOLO_WEIGHTS_PATH.exists() else None


# =========================================================================
# 1. PYDANTIC SCHEMA & GEMINI VISION PROMPT
# =========================================================================
class LocomotiveVisionDetection(BaseModel):
    is_train_or_locomotive_present: bool = Field(
        description="True if an engine or train locomotive is visible."
    )
    detected_loco_number: Optional[str] = Field(
        None,
        description="4-digit locomotive serial number in English digits (e.g. '2901', '2936', '3026'). Null if unreadable."
    )
    reading_location: Optional[str] = Field(
        None,
        description="'top_headlight_board', 'lower_buffer_beam', 'cab_side', or 'none'."
    )
    unreadable_reason: Optional[str] = Field(
        None,
        description="If null: 'occluded_by_passengers', 'blurry', 'too_far', or 'no_number_visible'."
    )
    is_at_station_platform: bool = Field(
        description="True if the locomotive is stopped at a passenger station/platform."
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0."
    )


VISION_SYSTEM_PROMPT = """
You are an expert railway vehicle identifier specializing in Bangladesh Railway (BR) diesel locomotives.
Your mission is to read the exact 4-digit locomotive road number from photographs or tight crops of train number plates.

==============================
1. BENGALI DIGIT TRANSLATION
==============================
Locomotive numbers in Bangladesh are painted in either English digits (e.g. 2901) or Bengali numerals (e.g. ২৯০১).
- Always translate Bengali numerals into standard English digits:
  ০ -> 0, ১ -> 1, ২ -> 2, ৩ -> 3, ৪ -> 4, ৫ -> 5, ৬ -> 6, ৭ -> 7, ৮ -> 8, ৯ -> 9.
- The returned `detected_loco_number` MUST ALWAYS be in 4 English digits (e.g. "2901", "2936").

==============================
2. WHERE TO LOOK ON BR LOCOMOTIVES
==============================
1. Top Cab Board: Rectangular illuminated box between or above the twin headlights (e.g. '2901').
2. Lower Buffer Beam: Stenciled in white/yellow on the lower red or black steel beam.
3. Cab Side panels.

==============================
3. VALID CLASSES
==============================
- Class 2600: 2600 to 2699
- Class 2700: 2700 to 2799
- Class 2900: 2900 to 2999
- Class 3000 / 3100+: 3000 to 3350

RULES:
- Return ONLY the 4-digit number.
- Do NOT confuse 3-digit train numbers (773, 774) with 4-digit engine numbers.
- Do NOT guess if blurry or occluded. Set detected_loco_number = null.
"""


# =========================================================================
# 2. YOLO PLATE CROPPING & CONTRAST ENHANCEMENT
# =========================================================================
def enhance_image_clahe(img_path: Path) -> Path:
    """Applies CLAHE contrast enhancement so faded paint numbers stand out."""
    enhanced_path = CROPS_DIR / f"enh_{img_path.name}"
    if enhanced_path.exists():
        return enhanced_path

    try:
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            return img_path

        lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

        cv2.imwrite(str(enhanced_path), enhanced)
        return enhanced_path
    except Exception:
        return img_path


def crop_plates_with_yolo(img_path: Path) -> List[Path]:
    """
    Uses your trained YOLO model to find and crop the number plate boxes.
    If YOLO is not yet trained or finds no boxes, falls back to the full image.
    """
    if not yolo_detector:
        return [img_path]

    crops = []
    try:
        cv_img = cv2.imread(str(img_path))
        if cv_img is None:
            return [img_path]

        results = yolo_detector.predict(cv_img, conf=0.30, verbose=False)

        for r_idx, r in enumerate(results):
            boxes = r.boxes.xyxy.cpu().numpy()
            for b_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                h, w = cv_img.shape[:2]

                # Add a 10% padding buffer around the plate box
                pad_x = int((x2 - x1) * 0.1)
                pad_y = int((y2 - y1) * 0.1)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)

                crop = cv_img[y1:y2, x1:x2]
                if crop.size > 0:
                    crop_file = CROPS_DIR / f"yolo_{img_path.stem}_b{r_idx}_{b_idx}.jpg"
                    cv2.imwrite(str(crop_file), crop)
                    crops.append(crop_file)

        return crops if crops else [img_path]
    except Exception as e:
        print(f"      [!] YOLO crop error on {img_path.name}: {e}", flush=True)
        return [img_path]


def analyze_locomotive_image(image_path: Path, max_retries: int = 3) -> Optional[LocomotiveVisionDetection]:
    """Sends enhanced image/crop to Gemini 3.5 Flash-Lite."""
    proc_path = enhance_image_clahe(image_path)

    try:
        pil_img = Image.open(proc_path).convert("RGB")
        if max(pil_img.size) > 1600:
            pil_img.thumbnail((1600, 1600))

        delay = 4
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=VISION_MODEL,
                    contents=[
                        pil_img,
                        "Inspect this Bangladesh Railway locomotive photo/crop. Read the 4-digit engine number in English or Bengali numerals."
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=LocomotiveVisionDetection,
                        system_instruction=VISION_SYSTEM_PROMPT,
                        temperature=0.0,
                    ),
                )
                time.sleep(4.2)  # Respect free tier 15 RPM
                return response.parsed
            except Exception as e:
                err = str(e)
                if any(c in err for c in ("503", "429", "UNAVAILABLE")):
                    time.sleep(delay)
                    delay = min(delay * 2, 25)
                else:
                    break
    except Exception as e:
        print(f"      [!] Vision error on {image_path.name}: {e}", flush=True)

    return None


def validate_br_loco(num_str: Optional[str]) -> Optional[str]:
    """Validates that the detected number is a valid BR locomotive series."""
    if not num_str:
        return None
    clean = str(num_str).translate(BN_TO_EN)
    m = re.search(r"\b(26\d{2}|27\d{2}|29\d{2}|30\d{2}|31\d{2}|32\d{2}|33\d{2})\b", clean)
    return m.group(1) if m else None


# =========================================================================
# 3. TEXT-FIRST EXTRACTION & BROWSER HARVESTING
# =========================================================================
def extract_kalni_locomotive_from_text(raw_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Checks if post text already specifies the loco (e.g. '2936/774' or '৭৭৩ ... ইঞ্জিন ২৯২১')."""
    if not raw_text:
        return None, None
    t = raw_text.translate(BN_TO_EN)

    m1 = re.search(r"\b([23]\d{3})\s*/\s*(773|774)\b", t)
    if m1:
        return m1.group(2), m1.group(1)

    m2 = re.search(r"\b(773|774)\s*/\s*([23]\d{3})\b", t)
    if m2:
        return m2.group(1), m2.group(2)

    for tr in ["773", "774"]:
        if tr in t:
            m3 = re.search(rf"{tr}.*?(?:ইঞ্জিন|লোকো|loco|engine|with|উইথ)?\s*[:=\-]?\s*([23]\d{{3}})", t, re.I | re.S)
            if m3:
                return tr, m3.group(1)

    return None, None


MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}


def parse_post_dt(s: str) -> Optional[datetime]:
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})", s)
    if not m:
        m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
        if not m:
            return None
        d, mon, y = m.groups()
        try:
            return datetime(int(y), MONTHS[mon.capitalize()], int(d), 12, 0)
        except Exception:
            return None
    d, mon, y, hh, mm = m.groups()
    try:
        return datetime(int(y), MONTHS[mon.capitalize()], int(d), int(hh), int(mm))
    except Exception:
        return None


def get_post_journey_and_train(p: dict) -> Tuple[Optional[str], Optional[str]]:
    ts = p.get("date_time", "")
    post_dt = parse_post_dt(ts)
    if not post_dt:
        return None, None

    txt = p.get("text", "").translate(BN_TO_EN)

    train = None
    if "773" in txt or "ঢাকা টু সিলেট" in txt or "সিলেটগামী" in txt or "লন্ডনগামী" in txt:
        train = "773"
    elif "774" in txt or "সিলেট টু ঢাকা" in txt or "ঢাকাগামী" in txt:
        train = "774"
    else:
        mins = post_dt.hour * 60 + post_dt.minute
        train = "774" if 360 <= mins < 810 else "773"

    cal_date = post_dt.date()
    if train == "773" and post_dt.hour < 6:
        cal_date = cal_date - timedelta(days=1)

    return cal_date.isoformat(), train


def is_port_listening(port=9222, host="127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def harvest_images_from_post(page, post_url: str, save_prefix: str) -> List[Path]:
    downloaded = []
    if not post_url or "facebook.com" not in post_url:
        return downloaded

    try:
        page.goto(post_url, wait_until="domcontentloaded", timeout=15000)
        page.wait_for_timeout(1000)

        img_urls = page.evaluate("""() => {
            const out = [];
            const imgs = document.querySelectorAll('div[data-ad-preview="message"] ~ div img, div[role="main"] img, img[data-visualcompletion="media-vc-image"]');
            for (const img of imgs) {
                const src = img.src || '';
                if (src && !src.includes('rsrc.php') && !src.includes('emoji.php') && (img.naturalWidth > 250 || img.width > 250)) {
                    out.push(src);
                }
            }
            return [...new Set(out)].slice(0, 3);
        }""")

        for idx, src in enumerate(img_urls):
            out_file = IMAGES_DIR / f"{save_prefix}_img{idx}.jpg"
            if out_file.exists():
                downloaded.append(out_file)
                continue

            try:
                resp = page.request.get(src, timeout=8000)
                if resp.status == 200:
                    out_file.write_bytes(resp.body())
                    downloaded.append(out_file)
            except Exception:
                pass
    except Exception as e:
        print(f"    [!] Error loading post {post_url}: {e}", flush=True)

    return downloaded


# =========================================================================
# 4. MASTER ENGINE PIPELINE
# =========================================================================
def run_engine_pipeline():
    if not JOURNEYS_CSV_FILE.exists() or not POSTS_JSON_FILE.exists():
        print(f"[-] Missing input files in {OUT_DIR}. Run build_dataset.py first.")
        return

    if not yolo_detector:
        print("[!] NOTICE: YOLO weights not found at runs/detect/loco_plate_detector/weights/best.pt.")
        print("[!] Run 'python train_yolo.py' first for maximum detection accuracy. Falling back to whole images...\n")
    else:
        print(f"[✔] Loaded trained YOLO plate detector from {YOLO_WEIGHTS_PATH}\n")

    journeys_df = pd.read_csv(JOURNEYS_CSV_FILE, dtype=str).fillna("")
    with open(POSTS_JSON_FILE, "r", encoding="utf-8") as f:
        posts = json.load(f)

    # 1. Map all posts by (journey_date, train_number)
    posts_by_journey = defaultdict(list)
    for p in posts:
        j_d, t_no = get_post_journey_and_train(p)
        if j_d and t_no:
            posts_by_journey[(j_d, t_no)].append(p)

    # 2. Filter journeys missing locomotive numbers
    missing_mask = (journeys_df["locomotive_number"] == "") | (journeys_df["locomotive_number"].str.lower() == "none")
    missing_journeys = journeys_df[missing_mask]

    print(f"[+] Total journeys: {len(journeys_df)} | Missing locomotive: {len(missing_journeys)}")
    if len(missing_journeys) == 0:
        print("[✓] All journeys already have locomotives!")
        return

    existing_detections = {}
    if ENGINE_CSV_FILE.exists():
        try:
            ex_df = pd.read_csv(ENGINE_CSV_FILE, dtype=str).fillna("")
            for r in ex_df.itertuples():
                existing_detections[(r.journey_date, r.train_number)] = r._asdict()
            print(f"[+] Loaded {len(existing_detections)} cached detections from {ENGINE_CSV_FILE}")
        except Exception:
            pass

    # =====================================================================
    # PHASE 1: TEXT-FIRST RESOLUTION (ZERO IMAGES DOWNLOADED)
    # =====================================================================
    unresolved_journeys = []
    text_resolved_count = 0

    print("\n[+] Phase 1: Checking post text for locomotive numbers first...", flush=True)

    for _, j_row in missing_journeys.iterrows():
        j_date = j_row["journey_date"]
        train_no = j_row["train_number"]
        key = (j_date, train_no)

        if key in existing_detections and existing_detections[key].get("loco_from_image"):
            continue

        candidate_posts = posts_by_journey.get(key, [])
        found_in_text = None

        for p in candidate_posts:
            tr_shorthand, loco_shorthand = extract_kalni_locomotive_from_text(p.get("text", ""))
            if tr_shorthand == train_no and loco_shorthand:
                valid_num = validate_br_loco(loco_shorthand)
                if valid_num:
                    found_in_text = (valid_num, p.get("url", ""))
                    break

        if found_in_text:
            loco_val, src_url = found_in_text
            existing_detections[key] = {
                "journey_date": j_date,
                "train_number": train_no,
                "loco_from_image": loco_val,
                "loco_image_votes": "1",
                "loco_image_score": "0.99",
                "loco_image_pre_departure": "1",
                "source_post_url": src_url
            }
            text_resolved_count += 1
            print(f"  [⚡ Text Resolved]: Journey {j_date}_{train_no} -> Loco {loco_val} (Skipped downloading images!)")
        else:
            unresolved_journeys.append(j_row)

    print(f"[✓] Phase 1 Complete: {text_resolved_count} journeys resolved directly from text.")
    print(f"[*] Remaining journeys requiring visual inspection: {len(unresolved_journeys)}\n")

    if not unresolved_journeys:
        pd.DataFrame(list(existing_detections.values())).to_csv(ENGINE_CSV_FILE, index=False, encoding="utf-8-sig")
        print(f"[🎉 DONE] All missing locomotives resolved. Saved to {ENGINE_CSV_FILE}")
        return

    # =====================================================================
    # PHASE 2: IMAGE HARVESTING + YOLO CROPPING + VISION ANALYSIS
    # =====================================================================
    snap_brave_bin = "/snap/bin/brave"
    snap_profile_dir = os.path.expanduser("~/snap/brave/current/.config/BraveSoftware/Brave-Browser")
    if not os.path.exists(snap_profile_dir):
        snap_profile_dir = os.path.expanduser("~/snap/brave/692/.config/BraveSoftware/Brave-Browser")

    with sync_playwright() as p:
        browser, context, page = None, None, None

        if is_port_listening(9222):
            print("[+] Attaching to Snap Brave on port 9222 over CDP...", flush=True)
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()
        else:
            print("[*] Launching Snap Brave directly...", flush=True)
            context = p.chromium.launch_persistent_context(
                user_data_dir=snap_profile_dir,
                executable_path=snap_brave_bin,
                headless=False,
                no_viewport=True,
            )
            page = context.pages[0] if context.pages else context.new_page()

        for j_row in unresolved_journeys:
            j_date = j_row["journey_date"]
            train_no = j_row["train_number"]
            key = (j_date, train_no)

            candidate_posts = posts_by_journey.get(key, [])
            if not candidate_posts:
                continue

            print(f"\n=======================================================")
            print(f"📸 Inspecting Images for {j_date} | Train {train_no} ({len(candidate_posts)} posts)")
            print(f"=======================================================")

            detected_numbers = []

            for p_idx, post_item in enumerate(candidate_posts):
                p_url = post_item.get("url")
                if not p_url:
                    continue

                prefix = f"{j_date}_{train_no}_post{p_idx}"
                image_files = harvest_images_from_post(page, p_url, prefix)

                for img_p in image_files:
                    # 1. Use YOLO to crop the small plate rectangle
                    plate_crops = crop_plates_with_yolo(img_p)

                    # 2. Inspect each magnified crop with Gemini Vision
                    for crop_p in plate_crops:
                        print(f"  🔍 Inspecting: {crop_p.name}...", flush=True)
                        res = analyze_locomotive_image(crop_p)

                        if res and res.is_train_or_locomotive_present and res.detected_loco_number:
                            valid_num = validate_br_loco(res.detected_loco_number)
                            if valid_num:
                                print(f"  [🎯 YOLO+VISION DETECTED]: {valid_num} (Conf: {res.confidence*100:.0f}%, Part: {res.reading_location})", flush=True)
                                detected_numbers.append({
                                    "loco": valid_num,
                                    "conf": res.confidence,
                                    "is_platform": res.is_at_station_platform,
                                    "url": p_url,
                                })

            # Consensus voting if detections exist
            if detected_numbers:
                best = max(detected_numbers, key=lambda x: x["conf"])
                row_entry = {
                    "journey_date": j_date,
                    "train_number": train_no,
                    "loco_from_image": best["loco"],
                    "loco_image_votes": str(len(detected_numbers)),
                    "loco_image_score": f"{best['conf']:.2f}",
                    "loco_image_pre_departure": "1" if best["is_platform"] else "0",
                    "source_post_url": best["url"]
                }
                existing_detections[key] = row_entry
                pd.DataFrame(list(existing_detections.values())).to_csv(ENGINE_CSV_FILE, index=False)
                print(f"  [✔] Assigned Loco {best['loco']} to Journey {j_date}_{train_no}!")

        if browser:
            browser.close()
        elif context:
            context.close()

    pd.DataFrame(list(existing_detections.values())).to_csv(ENGINE_CSV_FILE, index=False, encoding="utf-8-sig")
    print(f"\n[🎉 SUCCESS] Pipeline Complete! Saved to {ENGINE_CSV_FILE}")


if __name__ == "__main__":
    run_engine_pipeline()