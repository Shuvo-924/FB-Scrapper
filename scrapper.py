import datetime
import json
import os
import re
import shutil
import socket
import subprocess
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

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TUNNEL_PROCESSES = []

IGNORE_TOOLTIPS = {
    "marketplace", "home", "video", "friends", "groups",
    "gaming", "menu", "notifications", "account", "reels",
    "facebook", "watch", "messages", "see your profile"
}


def cleanup_tunnels():
    global TUNNEL_PROCESSES
    print("[*] Cleaning up live handover tunnel processes...", flush=True)
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
    """Launches x11vnc, websockify (noVNC), and Cloudflare Tunnel for live browser access."""
    try:
        print("\n🌐 [LIVE HANDOVER] Initializing VNC & Cloudflare Tunnel...", flush=True)
        subprocess.run(["pkill", "-9", "-f", "x11vnc"], capture_output=True)
        subprocess.run(["pkill", "-9", "-f", "websockify"], capture_output=True)
        time.sleep(1)

        display = os.environ.get("DISPLAY")
        if not display and os.path.exists("/tmp/.X11-unix"):
            for entry in sorted(os.listdir("/tmp/.X11-unix")):
                if entry.startswith("X"):
                    display = f":{entry[1:]}"
                    break
        if not display:
            display = ":99"

        print(f"🖥️ Target Virtual Display: {display}", flush=True)
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
                print("✅ x11vnc is active on port 5900!", flush=True)
                break
            if vnc_proc.poll() is not None:
                out, _ = vnc_proc.communicate()
                print(f"❌ x11vnc failed to start:\n{out}", flush=True)
                return False
            time.sleep(0.5)

        if not vnc_ready:
            print("❌ Timed out waiting for x11vnc.", flush=True)
            return False

        novnc_dir = "/usr/share/novnc"
        web_proc = subprocess.Popen(
            ["websockify", f"--web={novnc_dir}", "6080", "127.0.0.1:5900"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        TUNNEL_PROCESSES.append(web_proc)

        web_ready = False
        for _ in range(15):
            if is_port_listening(6080):
                web_ready = True
                print("✅ websockify is active on port 6080!", flush=True)
                break
            if web_proc.poll() is not None:
                out, _ = web_proc.communicate()
                print(f"❌ websockify failed to start:\n{out}", flush=True)
                return False
            time.sleep(0.5)

        if not web_ready:
            print("❌ Timed out waiting for websockify.", flush=True)
            return False

        # Prepare cloudflared binary
        cloudflared_bin = shutil.which("cloudflared") or "/usr/local/bin/cloudflared"
        if not os.path.exists(cloudflared_bin):
            tmp_cf = "/tmp/cloudflared"
            if not os.path.exists(tmp_cf):
                print("⬇️ Downloading cloudflared binary...", flush=True)
                urllib.request.urlretrieve(
                    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
                    tmp_cf
                )
                os.chmod(tmp_cf, 0o755)
            cloudflared_bin = tmp_cf

        cf_log_path = "/tmp/cloudflared.log"
        if os.path.exists(cf_log_path):
            os.remove(cf_log_path)

        cf_proc = subprocess.Popen(
            [cloudflared_bin, "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:6080", "--logfile", cf_log_path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        TUNNEL_PROCESSES.append(cf_proc)

        base_url = None
        start_wait = time.time()
        print("⏳ Generating Cloudflare Tunnel URL...", flush=True)

        while time.time() - start_wait < 35:
            if os.path.exists(cf_log_path):
                try:
                    with open(cf_log_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                        match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
                        if match:
                            base_url = match.group(0)
                            break
                except Exception:
                    pass
            time.sleep(1)

        if not base_url:
            print("❌ Timed out waiting for Cloudflare Tunnel URL.", flush=True)
            return False

        tunnel_url = (
            f"{base_url}/vnc.html?autoconnect=true&resize=scale&path=websockify&quality=5&compression=8&show_dot=true"
        )

        print("\n" + "=" * 75, flush=True)
        print("🚨 ACTION REQUIRED: LIVE BROWSER ACCESS IS READY!", flush=True)
        print(f"\n👉 OPEN THIS LINK IN YOUR BROWSER:\n   {tunnel_url}\n", flush=True)
        print("Log in to Facebook, complete 2FA/approval, and navigate to KalniBR.", flush=True)
        print("Once the timeline is visible, the script will automatically take over!", flush=True)
        print("=" * 75 + "\n", flush=True)
        return True
    except Exception as e:
        print(f"❌ Error setting up live handover: {e}", flush=True)
        return False


def wait_for_user_handover(page, context, timeout_minutes=10) -> bool:
    """Waits for user to complete login in the live browser."""
    if not start_live_handover():
        return False

    print(f"⏳ Live session active. Waiting up to {timeout_minutes} minutes for login...", flush=True)
    start_time = time.time()
    max_seconds = timeout_minutes * 60

    while time.time() - start_time < max_seconds:
        current_url = page.url

        # Check if user reached target page or completed login
        is_on_target = "KalniBR" in current_url
        is_auth_page = any(m in current_url for m in ["login", "checkpoint", "auth_platform"])
        
        # Check if feed cards are accessible
        feed_cards = get_feed_cards(page)

        if (is_on_target and feed_cards.count() > 0) or (not is_auth_page and "facebook.com" in current_url and "/login" not in current_url and feed_cards.count() > 0):
            print("\n" + "=" * 75, flush=True)
            print("🎉 [LOGIN CONFIRMED] KalniBR feed is visible!", flush=True)
            print("Relinquishing live VNC session and resuming autonomous scraping...", flush=True)
            print("=" * 75 + "\n", flush=True)
            
            # Save session for convenience
            try:
                context.storage_state(path="storage_state.json")
                print("[+] Saved session to storage_state.json", flush=True)
            except Exception:
                pass

            cleanup_tunnels()
            return True

        time.sleep(3)

    print("❌ Timed out waiting for manual login.", flush=True)
    cleanup_tunnels()
    return False


def matches_keywords(text: str) -> bool:
    if not text:
        return False
    return any(word in text for word in KEYWORDS)


def clean_text(text: str) -> str:
    text = re.sub(r"\b(See more|See less|আরও দেখুন|কম দেখুন)\b", "", text, flags=re.I)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_facebook_url(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    clean_params = []
    for k, v in parse_qs(parsed.query).items():
        if not k.startswith("__") and k != "comment_id":
            for val in v:
                clean_params.append(f"{k}={val}")
    query = f"?{'&'.join(clean_params)}" if clean_params else ""
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}{query}"


def is_valid_timestamp_tooltip(text: str) -> bool:
    if not text:
        return False
    lower = text.strip().lower()
    if lower in IGNORE_TOOLTIPS:
        return False
    has_digit = bool(re.search(r"[\d০-৯]", text))
    has_time_marker = bool(re.search(r"(:|at|ago|am|pm|মিনিট|ঘণ্টা|দিন|সপ্তাহ|বছর)", text, re.I))
    return has_digit or has_time_marker


def get_post_permalink(card) -> str:
    candidate_links = card.locator(
        'a[href*="/posts/"], '
        'a[href*="/videos/"], '
        'a[href*="/watch"], '
        'a[href*="/reel/"], '
        'a[href*="/photo"], '
        'a[href*="/permalink/"], '
        'a[href*="story_fbid="], '
        'a[href*="fbid="]'
    )

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

    header_time_link = card.locator(
        'div[data-ad-rendering-role="profile_name"] ~ div a[role="link"], '
        'a[target="_blank"]:has(svg)'
    )
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
        post_page.goto(post_url, wait_until="domcontentloaded", timeout=25000)
        try:
            post_page.wait_for_selector('[data-ad-rendering-role="profile_name"], h2, div[role="main"]', timeout=8000)
        except Exception:
            pass

        post_page.wait_for_timeout(1000)
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
            post_page.wait_for_timeout(300)
            box = target.bounding_box()
            if box:
                post_page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            else:
                target.hover(force=True, timeout=2000)

            post_page.wait_for_timeout(1200)
            tooltips = post_page.locator('div[role="tooltip"]')
            for idx in range(tooltips.count()):
                t_val = tooltips.nth(idx).inner_text().strip()
                if is_valid_timestamp_tooltip(t_val):
                    exact_date = t_val
                    break

        if not exact_date:
            html = post_page.content()
            match = re.search(r'"(?:creation_time|publish_time)":\s*(\d{10})', html)
            if match:
                unix_ts = int(match.group(1))
                dt = datetime.datetime.fromtimestamp(unix_ts)
                exact_date = dt.strftime("%A, %d %B %Y at %H:%M")

        return exact_date
    except Exception as e:
        print(f"    [!] Timestamp parse error: {e}", flush=True)
        return ""
    finally:
        if post_page:
            try:
                post_page.close()
            except Exception:
                pass


def save_data(collected_posts: dict):
    if not collected_posts:
        return
    df = pd.DataFrame(list(collected_posts.values()))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_json(OUTPUT_DIR / "posts.json", orient="records", force_ascii=False, indent=2)
    df.to_csv(OUTPUT_DIR / "posts.csv", index=False, encoding="utf-8-sig")


def main():
    collected_posts = {}

    with sync_playwright() as p:
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

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        print(f"[+] Navigating to: {PAGE_URL}", flush=True)
        page.goto("https://facebook.com", wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(4000)


        wait_for_user_handover(page, context, timeout_minutes=10)

        # Ensure we are on KalniBR and the feed is visible
        if "KalniBR" not in page.url:
            print(f"[+] Directing to {PAGE_URL}...", flush=True)
            page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=45000)
            page.wait_for_timeout(4000)

        unlock_facebook_scroll(page)
        viewport_h = 1080

        print(f"[+] Starting pass 1 scraping across up to {MAX_SCROLLS} scrolls...", flush=True)

        last_height = 0
        stalled_count = 0

        for scroll_no in range(MAX_SCROLLS):
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
                    post_text = ""
                    if msg_loc.count() > 0:
                        post_text = clean_text(msg_loc.first.inner_text(timeout=400))
                    else:
                        colored_loc = card.locator('div[style*="font-size: 30px"], div[style*="font-size: 24px"]')
                        if colored_loc.count() > 0:
                            post_text = clean_text(colored_loc.first.inner_text(timeout=300))

                    if not matches_keywords(post_text):
                        continue

                    url = get_post_permalink(card)
                    sig = url or post_text[:100]

                    if sig in collected_posts:
                        continue

                    try:
                        see_more = card.locator('div[role="button"]:has-text("See more"), div[role="button"]:has-text("আরও দেখুন")')
                        if see_more.count() > 0 and see_more.first.is_visible():
                            see_more.first.click(timeout=300, force=True)
                            post_text = clean_text(msg_loc.first.inner_text(timeout=400))
                    except Exception:
                        pass

                    collected_posts[sig] = {
                        "date_time": "",
                        "text": post_text,
                        "url": url,
                    }
                    print(f"  [Found Post] Matched: {[w for w in KEYWORDS if w in post_text]} | URL: {url or 'N/A'}", flush=True)

                except Exception:
                    continue

            # Verify scroll progression
            current_height = page.evaluate("document.body.scrollHeight")
            if current_height == last_height:
                stalled_count += 1
                if stalled_count >= 4:
                    print(f"  [!] Feed stalled at {current_height}px. Nudging down...", flush=True)
                    page.keyboard.press("PageDown")
                    page.wait_for_timeout(1000)
                    page.keyboard.press("End")
                    page.wait_for_timeout(1500)
                    if stalled_count == 12:
                        page.screenshot(path="output/debug_feed.png")
            else:
                stalled_count = 0
                last_height = current_height

            page.keyboard.press("PageDown")
            page.wait_for_timeout(SCROLL_WAIT_MS)

            if (scroll_no + 1) % 25 == 0 or scroll_no == MAX_SCROLLS - 1:
                save_data(collected_posts)
                print(f"Scroll {scroll_no + 1}/{MAX_SCROLLS} | Matched: {len(collected_posts)} | Height: {current_height}px", flush=True)

        print(f"\n[+] Pass 1 complete. Extracting timestamps for {len(collected_posts)} matched posts...", flush=True)
        processed = 0
        for sig, post in collected_posts.items():
            if post["url"] and not post["date_time"]:
                print(f"  Fetching date for: {post['url']}", flush=True)
                exact_ts = scrape_post_timestamp_from_url(context, post["url"])
                post["date_time"] = exact_ts
                print(f"  -> Date: {exact_ts or 'Not available'}", flush=True)
                processed += 1
                if processed % 10 == 0:
                    save_data(collected_posts)

        save_data(collected_posts)
        print(f"\nDone! Exported {len(collected_posts)} posts to output/.", flush=True)
        cleanup_tunnels()
        browser.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        cleanup_tunnels()