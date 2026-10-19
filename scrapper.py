import datetime
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = os.getenv("PAGE_URL", "https://www.facebook.com/KalniBR")
MAX_SCROLLS = int(os.getenv("MAX_SCROLLS", "1500"))
SCROLL_WAIT_MS = int(os.getenv("SCROLL_WAIT_MS", "2500"))
KEYWORDS_ENV = os.getenv("KEYWORDS", "ঢাকা,সিলেট,কমলাপুর")
KEYWORDS = [k.strip() for k in KEYWORDS_ENV.split(",") if k.strip()]

# Maximum time to run in this single job before chaining to the next (e.g. 325 mins)
MAX_RUNTIME_MINUTES = float(os.getenv("MAX_RUNTIME_MINUTES", "325"))
MAX_RUNTIME_SECONDS = MAX_RUNTIME_MINUTES * 60

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = Path("checkpoint.json")

TUNNEL_PROCESSES = []
COLLECTED_POSTS = {}

IGNORE_TOOLTIPS = {
    "marketplace", "home", "video", "friends", "groups",
    "gaming", "menu", "notifications", "account", "reels",
    "facebook", "watch", "messages", "see your profile"
}


def save_data():
    global COLLECTED_POSTS
    if not COLLECTED_POSTS:
        return
    try:
        df = pd.DataFrame(list(COLLECTED_POSTS.values()))
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        df.to_json(OUTPUT_DIR / "posts.json", orient="records", force_ascii=False, indent=2)
        df.to_csv(OUTPUT_DIR / "posts.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[!] Error auto-saving data: {e}", flush=True)


def save_checkpoint_and_trigger(scrolls_completed: int, total_desired: int, last_height: int, context):
    """Saves session cookies, checkpoint progress, and marks trigger for next workflow."""
    print("\n[+] Creating checkpoint for next GitHub Actions run...", flush=True)
    
    # 1. Save session cookies so the next run NEVER asks for login/VNC
    try:
        context.storage_state(path="storage_state.json")
        print("  [✓] Updated storage_state.json", flush=True)
    except Exception as e:
        print(f"  [!] Failed saving storage_state: {e}", flush=True)

    # 2. Save data
    save_data()

    remaining_scrolls = max(0, total_desired - scrolls_completed)
    
    checkpoint_data = {
        "timestamp": datetime.datetime.now().isoformat(),
        "scrolls_completed": scrolls_completed,
        "remaining_scrolls": remaining_scrolls,
        "last_height": last_height,
        "total_posts_found": len(COLLECTED_POSTS)
    }
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump(checkpoint_data, f, indent=2)

    # 3. If there are still scrolls remaining, write trigger file for GitHub Actions
    if remaining_scrolls > 0:
        with open("trigger_next.txt", "w", encoding="utf-8") as f:
            f.write(str(remaining_scrolls))
        print(f"  [✓] Created trigger_next.txt with {remaining_scrolls} remaining scrolls.", flush=True)
    else:
        if os.path.exists("trigger_next.txt"):
            os.remove("trigger_next.txt")
        print("  [✓] Goal achieved! No further runs needed.", flush=True)


def handle_shutdown_signal(signum, frame):
    print(f"\n[⚠️] Shutdown signal ({signum}) received! Flushing data...", flush=True)
    save_data()
    cleanup_tunnels()
    sys.exit(0)


signal.signal(signal.SIGTERM, handle_shutdown_signal)
signal.signal(signal.SIGINT, handle_shutdown_signal)


def cleanup_tunnels():
    global TUNNEL_PROCESSES
    for p in TUNNEL_PROCESSES:
        try:
            p.terminate()
            p.wait(timeout=2)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    TUNNEL_PROCESSES = []


def is_port_listening(port, host="127.0.0.1", timeout=1.0):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def start_live_handover() -> bool:
    try:
        print("\n🌐 [LIVE HANDOVER] Initializing VNC & Multi-Tunnel...", flush=True)
        subprocess.run(["pkill", "-9", "-f", "x11vnc"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "websockify"], capture_output=True)
        time.sleep(1)

        display = os.environ.get("DISPLAY", ":99")
        x11vnc_cmd = [
            "x11vnc", "-display", display, "-nopw", "-listen", "127.0.0.1",
            "-rfbport", "5900", "-xdamage", "-defer", "15", "-wait", "20",
            "-xkb", "-forever", "-shared",
        ]
        xauth = os.environ.get("XAUTHORITY")
        if xauth and os.path.exists(xauth):
            x11vnc_cmd.extend(["-auth", xauth])

        vnc_proc = subprocess.Popen(x11vnc_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        TUNNEL_PROCESSES.append(vnc_proc)

        vnc_ready = False
        for _ in range(15):
            if is_port_listening(5900):
                vnc_ready = True
                break
            time.sleep(0.5)

        if not vnc_ready:
            return False

        web_proc = subprocess.Popen(
            ["websockify", "--web=/usr/share/novnc", "6080", "127.0.0.1:5900"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        TUNNEL_PROCESSES.append(web_proc)

        # Cloudflare Tunnel
        cloudflared_bin = shutil.which("cloudflared") or "/tmp/cloudflared"
        if not os.path.exists(cloudflared_bin):
            urllib.request.urlretrieve(
                "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                "/tmp/cloudflared"
            )
            os.chmod("/tmp/cloudflared", 0o755)
            cloudflared_bin = "/tmp/cloudflared"

        cf_log_path = "/tmp/cloudflared.log"
        if os.path.exists(cf_log_path):
            os.remove(cf_log_path)

        cf_proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "--protocol", "http2", "--no-autoupdate", "--url", "http://127.0.0.1:6080", "--logfile", cf_log_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        TUNNEL_PROCESSES.append(cf_proc)

        # Pinggy Tunnel
        pinggy_base_url = None
        pinggy_proc = subprocess.Popen(
            ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "ServerAliveInterval=30", "-p", "443", "-R0:localhost:6080", "a.pinggy.io"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        TUNNEL_PROCESSES.append(pinggy_proc)

        def read_pinggy():
            nonlocal pinggy_base_url
            try:
                for line in iter(pinggy_proc.stdout.readline, ''):
                    match = re.search(r"https://[a-zA-Z0-9-]+\.a\.pinggy\.link", line)
                    if match:
                        pinggy_base_url = match.group(0)
                        break
            except Exception:
                pass

        threading.Thread(target=read_pinggy, daemon=True).start()

        cf_base_url = None
        start_wait = time.time()
        while time.time() - start_wait < 25:
            if not cf_base_url and os.path.exists(cf_log_path):
                try:
                    with open(cf_log_path, "r", encoding="utf-8", errors="ignore") as f:
                        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", f.read())
                        if match:
                            cf_base_url = match.group(0)
                except Exception:
                    pass

            if cf_base_url or pinggy_base_url:
                break
            time.sleep(1)

        vnc_suffix = "/vnc.html?autoconnect=true&resize=scale&path=websockify&quality=5&compression=8&show_dot=true"
        print("\n" + "=" * 75, flush=True)
        print("🚨 ACTION REQUIRED: LIVE BROWSER ACCESS IS READY!", flush=True)
        if cf_base_url:
            print(f"👉 PRIMARY LINK (Cloudflare): {cf_base_url}{vnc_suffix}", flush=True)
        if pinggy_base_url:
            print(f"👉 BACKUP LINK (Pinggy):     {pinggy_base_url}{vnc_suffix}", flush=True)
        print("=" * 75 + "\n", flush=True)
        return bool(cf_base_url or pinggy_base_url)
    except Exception as e:
        print(f"❌ Error setting up live handover: {e}", flush=True)
        return False


def wait_for_user_handover(page, context, timeout_minutes=10) -> bool:
    if not start_live_handover():
        return False
    print(f"⏳ Waiting up to {timeout_minutes} minutes for login...", flush=True)
    start_time = time.time()
    while time.time() - start_time < (timeout_minutes * 60):
        current_url = page.url
        is_on_target = "KalniBR" in current_url
        is_auth_page = any(m in current_url for m in ["login", "checkpoint", "auth_platform"])
        feed_cards = get_feed_cards(page)

        if (is_on_target and feed_cards.count() > 0) or (not is_auth_page and "facebook.com" in current_url and "/login" not in current_url and feed_cards.count() > 0):
            print("\n🎉 [LOGIN CONFIRMED] KalniBR feed visible! Resuming scraper...", flush=True)
            try:
                context.storage_state(path="storage_state.json")
            except Exception:
                pass
            cleanup_tunnels()
            return True
        time.sleep(3)
    cleanup_tunnels()
    return False


def matches_keywords(text: str) -> bool:
    return any(word in text for word in KEYWORDS) if text else False


def clean_text(text: str) -> str:
    text = re.sub(r"\b(See more|See less|আরও দেখুন|কম দেখুন)\b", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_facebook_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    clean_params = [f"{k}={val}" for k, v in parse_qs(parsed.query).items() if not k.startswith("__") and k != "comment_id" for val in v]
    query = f"?{'&'.join(clean_params)}" if clean_params else ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query}"


def is_valid_timestamp_tooltip(text: str) -> bool:
    if not text or text.strip().lower() in IGNORE_TOOLTIPS:
        return False
    return bool(re.search(r"[\d০-৯]", text)) or bool(re.search(r"(:|at|ago|am|pm|মিনিট|ঘণ্টা|দিন|সপ্তাহ|বছর)", text, re.I))


def get_post_permalink(card) -> str:
    candidate_links = card.locator('a[href*="/posts/"], a[href*="/videos/"], a[href*="/watch"], a[href*="/reel/"], a[href*="/photo"], a[href*="/permalink/"], a[href*="story_fbid="], a[href*="fbid="]')
    for i in range(candidate_links.count()):
        try:
            raw_href = candidate_links.nth(i).get_attribute("href") or ""
            parsed = urlparse(raw_href)
            if any(marker in parsed.path for marker in ["/posts/", "/videos/", "/watch", "/reel/", "/photo", "/permalink/"]):
                return clean_facebook_url(urljoin("https://www.facebook.com", raw_href))
            if any(param in parsed.query for param in ["story_fbid", "fbid", "v="]):
                return clean_facebook_url(urljoin("https://www.facebook.com", raw_href))
        except Exception:
            continue
    header_time_link = card.locator('div[data-ad-rendering-role="profile_name"] ~ div a[role="link"], a[target="_blank"]:has(svg)')
    if header_time_link.count() > 0:
        try:
            raw_href = header_time_link.first.get_attribute("href") or ""
            if raw_href and not raw_href.startswith("#"):
                return clean_facebook_url(urljoin("https://www.facebook.com", raw_href))
        except Exception:
            pass
    return ""


def unlock_facebook_scroll(page):
    try:
        page.evaluate("""() => {
            document.documentElement.style.overflow = 'auto';
            document.body.style.overflow = 'auto';
            const overlays = document.querySelectorAll('div[data-nosnippet], div[role="dialog"]');
            overlays.forEach(el => {
                if (el.innerText.includes("Log In") || el.innerText.includes("See more on Facebook") || el.innerText.includes("লগ ইন করুন")) {
                    el.remove();
                }
            });
        }""")
    except Exception:
        pass


def get_feed_cards(page):
    selectors = [
        'div[role="feed"] > div:has(div[data-ad-preview="message"])',
        'div[role="feed"] > div:has(div[data-ad-rendering-role="story_message"])',
        'div[style*="var(--card-corner-radius)"]:has(div[data-ad-rendering-role="profile_name"])',
        'div[style*="var(--card-corner-radius)"]:has(div[aria-label*="Actions for this post"])',
        'div[role="feed"] > div'
    ]
    for sel in selectors:
        cards = page.locator(sel)
        if cards.count() > 0:
            return cards
    return page.locator('div:has(> div [data-ad-rendering-role="profile_name"])')


def scrape_post_timestamp_from_url(context, post_url: str) -> str:
    if not post_url:
        return ""
    post_page = None
    try:
        post_page = context.new_page()
        post_page.goto(post_url, wait_until="domcontentloaded", timeout=20000)
        try:
            post_page.wait_for_selector('[data-ad-rendering-role="profile_name"], h2, div[role="main"]', timeout=6000)
        except Exception:
            pass
        post_page.wait_for_timeout(800)
        timestamp_selectors = [
            'div[data-ad-rendering-role="profile_name"] ~ div a[role="link"]',
            'h2 ~ div a[role="link"]',
            'a:has(use[*|href*="SvgT"]), a:has(use[href*="SvgT"])',
            'a[target="_blank"][href*="__cft__"]',
            'a[role="link"]:has(svg)'
        ]
        target = None
        for sel in timestamp_selectors:
            loc = post_page.locator(sel)
            if loc.count() > 0:
                for i in range(loc.count()):
                    cand = loc.nth(i)
                    if cand.is_visible():
                        target = cand
                        break
            if target:
                break
        exact_date = ""
        if target:
            target.evaluate("el => el.scrollIntoView({ block: 'center', inline: 'center' })")
            post_page.wait_for_timeout(200)
            box = target.bounding_box()
            if box:
                post_page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            else:
                target.hover(force=True, timeout=1500)
            post_page.wait_for_timeout(1000)
            tooltips = post_page.locator('div[role="tooltip"]')
            for idx in range(tooltips.count()):
                t_val = tooltips.nth(idx).inner_text().strip()
                if is_valid_timestamp_tooltip(t_val):
                    exact_date = t_val
                    break
        if not exact_date:
            match = re.search(r'"(?:creation_time|publish_time)":\s*(\d{10})', post_page.content())
            if match:
                exact_date = datetime.datetime.fromtimestamp(int(match.group(1))).strftime("%A, %d %B %Y at %H:%M")
        return exact_date
    except Exception as e:
        print(f"      [!] Timestamp extraction error: {e}", flush=True)
        return ""
    finally:
        if post_page:
            try:
                post_page.close()
            except Exception:
                pass


def main():
    global COLLECTED_POSTS
    start_job_time = time.time()

    # 1. Load already scraped posts from previous runs
    existing_file = OUTPUT_DIR / "posts.json"
    if existing_file.exists():
        try:
            with open(existing_file, "r", encoding="utf-8") as f:
                for item in json.load(f):
                    sig = item.get("url") or item.get("text", "")[:100]
                    COLLECTED_POSTS[sig] = item
            print(f"[+] Loaded {len(COLLECTED_POSTS)} posts from previous runs.", flush=True)
        except Exception:
            pass

    # 2. Check if resuming from an earlier checkpoint
    resume_height = 0
    if CHECKPOINT_FILE.exists():
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                chk = json.load(f)
                resume_height = chk.get("last_height", 0)
                print(f"[+] Found checkpoint. Last height: {resume_height}px. Completed: {chk.get('scrolls_completed', 0)} scrolls.", flush=True)
        except Exception:
            pass

    with sync_playwright() as p:
        storage_state = "storage_state.json" if Path("storage_state.json").exists() else None

        browser = p.chromium.launch(
            headless=False,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--start-maximized",
            ],
        )

        context_kwargs = {
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        }
        if storage_state:
            print("[+] Using stored session cookies (storage_state.json) - skipping login!", flush=True)
            context_kwargs["storage_state"] = storage_state

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        print(f"[+] Navigating to: {PAGE_URL}", flush=True)
        page.goto("https://www.facebook.com", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(4000)

        is_auth_page = any(m in page.url for m in ["login", "checkpoint", "auth_platform"])
        cards = get_feed_cards(page)

        if is_auth_page or cards.count() == 0:
            print("[!] Interaction or login required. Opening Live Handover...", flush=True)
            wait_for_user_handover(page, context, timeout_minutes=10)

        if "KalniBR" not in page.url:
            page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(4000)

        unlock_facebook_scroll(page)
        viewport_h = 1080

        # Fast-forward scroll if resuming from a previous run
        if resume_height > 1500:
            print(f"[+] Fast-forwarding feed to {resume_height}px...", flush=True)
            page.evaluate(f"window.scrollTo(0, {resume_height})")
            page.wait_for_timeout(3000)

        print(f"[+] Running scraper for up to {MAX_SCROLLS} scrolls (Max time budget: {MAX_RUNTIME_MINUTES}m)...", flush=True)

        last_height = 0
        stalled_count = 0

        for scroll_no in range(MAX_SCROLLS):
            # -------------------------------------------------------------
            # TIME-BUDGET CHECK: Hand off cleanly before GitHub kills VM
            # -------------------------------------------------------------
            elapsed = time.time() - start_job_time
            if elapsed >= MAX_RUNTIME_SECONDS:
                print(f"\n[⏱️] Reached time limit ({MAX_RUNTIME_MINUTES} mins). Saving progress to chain next run...", flush=True)
                save_checkpoint_and_trigger(scroll_no + 1, MAX_SCROLLS, last_height, context)
                cleanup_tunnels()
                browser.close()
                return

            unlock_facebook_scroll(page)
            cards = get_feed_cards(page)
            card_count = cards.count()

            for i in range(card_count):
                card = cards.nth(i)
                try:
                    box = card.bounding_box()
                    if not box or (box["y"] + box["height"]) < 60 or box["y"] > (viewport_h + 400):
                        continue

                    msg_loc = card.locator('div[data-ad-preview="message"], div[data-ad-rendering-role="story_message"]')
                    post_text = clean_text(msg_loc.first.inner_text(timeout=300)) if msg_loc.count() > 0 else ""
                    if not post_text:
                        colored_loc = card.locator('div[style*="font-size: 30px"], div[style*="font-size: 24px"]')
                        if colored_loc.count() > 0:
                            post_text = clean_text(colored_loc.first.inner_text(timeout=300))

                    if not matches_keywords(post_text):
                        continue

                    url = get_post_permalink(card)
                    sig = url or post_text[:100]

                    if sig in COLLECTED_POSTS:
                        continue

                    try:
                        see_more = card.locator('div[role="button"]:has-text("See more"), div[role="button"]:has-text("আরও দেখুন")')
                        if see_more.count() > 0 and see_more.first.is_visible():
                            see_more.first.click(timeout=250, force=True)
                            post_text = clean_text(msg_loc.first.inner_text(timeout=350))
                    except Exception:
                        pass

                    exact_ts = scrape_post_timestamp_from_url(context, url) if url else ""
                    COLLECTED_POSTS[sig] = {
                        "date_time": exact_ts,
                        "text": post_text,
                        "url": url,
                    }
                    save_data()
                    print(f"  [🎯 Matched #{len(COLLECTED_POSTS)}] {exact_ts or 'No Date'} | URL: {url or 'N/A'}", flush=True)
                except Exception:
                    continue

            # Adaptive scroll & stall recovery
            current_height = page.evaluate("() => Math.max(document.body.scrollHeight, document.documentElement.scrollHeight)")
            if current_height == last_height:
                stalled_count += 1
                page.keyboard.press("Escape")
                page.wait_for_timeout(200)

                if stalled_count >= 2:
                    page.evaluate("window.scrollBy(0, -800)")
                    page.wait_for_timeout(800)
                    page.evaluate("window.scrollTo(0, document.documentElement.scrollHeight)")
                    page.wait_for_timeout(1200)
                    try:
                        feed = page.locator('div[role="feed"]').first
                        if feed.is_visible():
                            feed.click(position={"x": 50, "y": 50}, force=True)
                    except Exception:
                        pass
                    page.keyboard.press("PageDown")
                    page.wait_for_timeout(1000)

                if stalled_count >= 12:
                    page.mouse.wheel(0, 3000)
                    page.wait_for_timeout(2000)
            else:
                stalled_count = 0
                last_height = current_height

            page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.9))")
            page.wait_for_timeout(SCROLL_WAIT_MS)

            if (scroll_no + 1) % 25 == 0:
                print(f"Scroll {scroll_no + 1}/{MAX_SCROLLS} | Posts: {len(COLLECTED_POSTS)} | Height: {current_height}px", flush=True)

        # Scrape loop finished normally
        save_checkpoint_and_trigger(MAX_SCROLLS, MAX_SCROLLS, last_height, context)
        cleanup_tunnels()
        browser.close()
        print(f"\n🎉 Finished all {MAX_SCROLLS} scrolls! Total posts: {len(COLLECTED_POSTS)}", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        save_data()
        cleanup_tunnels()