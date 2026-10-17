import argparse
import datetime
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = os.getenv("PAGE_URL", "https://www.facebook.com/KalniBR")
MAX_SCROLLS = int(os.getenv("MAX_SCROLLS", "500"))
SCROLL_WAIT_MS = int(os.getenv("SCROLL_WAIT_MS", "2500"))
KEYWORDS_ENV = os.getenv("KEYWORDS", "ঢাকা,সিলেট,কমলাপুর")
KEYWORDS = [k.strip() for k in KEYWORDS_ENV.split(",") if k.strip()]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IGNORE_TOOLTIPS = {
    "marketplace", "home", "video", "friends", "groups",
    "gaming", "menu", "notifications", "account", "reels",
    "facebook", "watch", "messages", "see your profile"
}


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


def dismiss_login_banner(page):
    """Dismisses Facebook's intrusive unauthenticated banners/overlays."""
    close_selectors = [
        'div[role="dialog"] div[aria-label="Close"]',
        'div[role="banner"] div[aria-label="Close"]',
        'div[aria-label="Decline optional cookies"]',
        'div[aria-label="Allow all cookies"]',
        'div[role="button"]:has-text("Not now")',
        'div[role="button"]:has-text("এখন নয়")'
    ]
    for selector in close_selectors:
        try:
            btn = page.locator(selector).first
            if btn.is_visible(timeout=1000):
                btn.click()
                page.wait_for_timeout(500)
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
            post_page.wait_for_selector(
                '[data-ad-rendering-role="profile_name"], h2, div[role="main"]',
                timeout=8000
            )
        except Exception:
            pass

        post_page.wait_for_timeout(1200)

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
        print(f"    [!] Timestamp parse error: {e}")
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
    json_path = OUTPUT_DIR / "posts.json"
    csv_path = OUTPUT_DIR / "posts.csv"
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


def main():
    collected_posts = {}
    is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"
    headless = True if is_ci else os.getenv("HEADLESS", "false").lower() == "true"

    # Load existing posts if resuming an earlier artifact
    existing_json = OUTPUT_DIR / "posts.json"
    if existing_json.exists():
        try:
            with open(existing_json, "r", encoding="utf-8") as f:
                prev_data = json.load(f)
                for item in prev_data:
                    sig = item.get("url") or item.get("text", "")[:100]
                    collected_posts[sig] = item
            print(f"[+] Loaded {len(collected_posts)} existing records from previous run.")
        except Exception:
            pass

    with sync_playwright() as p:
        storage_state = "storage_state.json" if Path("storage_state.json").exists() else None

        browser = p.chromium.launch(
            headless=headless,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )

        context_kwargs = {
            "viewport": {"width": 1366, "height": 900},
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        if storage_state:
            print("[+] Using stored session cookies (storage_state.json)")
            context_kwargs["storage_state"] = storage_state

        context = browser.new_context(**context_kwargs)
        page = context.new_page()

        print(f"[+] Navigating to: {PAGE_URL}")
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(3000)

        dismiss_login_banner(page)
        viewport_h = page.viewport_size["height"] if page.viewport_size else 900

        print(f"[+] Starting pass 1 scraping across up to {MAX_SCROLLS} scrolls...")

        for scroll_no in range(MAX_SCROLLS):
            if scroll_no % 15 == 0:
                dismiss_login_banner(page)

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
                    print(f"  [Post Found] {url or post_text[:40]}...")

                except Exception:
                    continue

            # Periodically write data to disk every 25 scrolls
            if (scroll_no + 1) % 25 == 0 or scroll_no == MAX_SCROLLS - 1:
                save_data(collected_posts)
                print(f"[Scroll {scroll_no + 1}/{MAX_SCROLLS}] Auto-saved. Unique matched: {len(collected_posts)}")

            page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.85))")
            page.wait_for_timeout(SCROLL_WAIT_MS)

        print(f"\n[+] Pass 1 complete. Extracting timestamps for {len(collected_posts)} posts...")
        processed = 0
        for sig, post in collected_posts.items():
            if post["url"] and not post["date_time"]:
                post["date_time"] = scrape_post_timestamp_from_url(context, post["url"])
                processed += 1
                if processed % 10 == 0:
                    save_data(collected_posts)

        save_data(collected_posts)
        print(f"\n[+] Finished. Total {len(collected_posts)} records written to output/.")
        browser.close()


if __name__ == "__main__":
    main()