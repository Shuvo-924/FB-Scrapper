import csv
import datetime
import json
import re
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = "https://www.facebook.com/KalniBR"

# Any of these keywords can be present for a valid post
KEYWORDS = ["ঢাকা", "সিলেট", "কমলাপুর"]

MAX_SCROLLS = 500
SCROLL_WAIT_MS = 2000

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
    """Strips tracking query parameters (__cft__, __tn__, comment_id)."""
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
    """Ensures tooltip is a real date/time and not a menu item."""
    if not text:
        return False
    lower = text.strip().lower()
    if lower in IGNORE_TOOLTIPS:
        return False
    has_digit = bool(re.search(r"[\d০-৯]", text))
    has_time_marker = bool(re.search(r"(:|at|ago|am|pm|মিনিট|ঘণ্টা|দিন|সপ্তাহ|বছর)", text, re.I))
    return has_digit or has_time_marker


def get_post_permalink(card) -> str:
    """
    Extracts the canonical link for ALL post types:
    - Standard posts (/posts/)
    - Videos (/videos/, /watch?v=)
    - Reels (/reel/)
    - Photos (/photo/, /photos/)
    - Permalinks (story_fbid, fbid, /permalink/)
    """
    # 1. Broad selector checking all Facebook content formats
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

            # Match valid post permalink paths
            if any(marker in parsed.path for marker in ["/posts/", "/videos/", "/watch", "/reel/", "/photo", "/permalink/"]):
                return clean_facebook_url(urljoin("https://www.facebook.com", raw_href))
            
            # Match query-based permalinks (e.g. ?story_fbid=... or ?fbid=...)
            if any(param in parsed.query for param in ["story_fbid", "fbid", "v="]):
                return clean_facebook_url(urljoin("https://www.facebook.com", raw_href))
        except Exception:
            continue

    # 2. Fallback: Check the header timestamp anchor directly
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


def get_feed_cards(page):
    """Modern Facebook Page post container selectors."""
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
    """
    Opens the post in a dedicated tab, waits for React hydration,
    and hovers directly over the timestamp link to read the tooltip.
    """
    if not post_url:
        return ""

    post_page = None
    try:
        post_page = context.new_page()
        # Navigate and wait for DOM
        post_page.goto(post_url, wait_until="domcontentloaded", timeout=20000)

        # Wait for the post header/main content to mount (allow React hydration)
        try:
            post_page.wait_for_selector(
                '[data-ad-rendering-role="profile_name"], h2, div[role="main"]',
                timeout=8000
            )
        except Exception:
            pass

        # Give React another moment to bind event listeners
        post_page.wait_for_timeout(1500)

        # In dedicated post view, find candidate timestamp anchors
        # The timestamp link is immediately below/next to the author name
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

        # Method 1: Hover over the found timestamp link
        if target:
            # Scroll gently into middle view
            target.evaluate("el => el.scrollIntoView({ block: 'center', inline: 'center' })")
            post_page.wait_for_timeout(400)

            box = target.bounding_box()
            if box:
                # Hover at the center of the timestamp
                post_page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
            else:
                target.hover(force=True, timeout=2000)

            # Wait for Facebook's 500ms hover delay
            post_page.wait_for_timeout(1200)

            # Read the tooltip
            tooltips = post_page.locator('div[role="tooltip"]')
            for idx in range(tooltips.count()):
                t_val = tooltips.nth(idx).inner_text().strip()
                if is_valid_timestamp_tooltip(t_val):
                    exact_date = t_val
                    break

        # Method 2 (Fallback): Extract creation_time from Facebook's embedded JSON
        if not exact_date:
            html = post_page.content()
            match = re.search(r'"(?:creation_time|publish_time)":\s*(\d{10})', html)
            if match:
                unix_ts = int(match.group(1))
                dt = datetime.datetime.fromtimestamp(unix_ts)
                exact_date = dt.strftime("%A, %d %B %Y at %H:%M")

        return exact_date

    except Exception as e:
        print(f"    [!] Error during timestamp extraction: {e}")
        return ""
    finally:
        if post_page:
            try:
                post_page.close()
            except Exception:
                pass


def main():
    collected_posts = {}

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir="facebook_profile",
            headless=False,
            viewport={"width": 1366, "height": 900},
        )

        page = context.pages[0] if context.pages else context.new_page()
        page.goto(PAGE_URL, wait_until="domcontentloaded")
        viewport_h = page.viewport_size["height"] if page.viewport_size else 900

        print("\n[+] Browser loaded.")
        print("[+] Ensure the Facebook Page timeline is visible.")
        input("[+] Press ENTER to start smooth scrolling...")

        # -------------------------------------------------------------
        # PASS 1: Smooth Downward Scroll (No hovering, No scroll jumps)
        # -------------------------------------------------------------
        for scroll_no in range(MAX_SCROLLS):
            cards = get_feed_cards(page)
            card_count = cards.count()

            for i in range(card_count):
                card = cards.nth(i)
                try:
                    box = card.bounding_box()
                    if not box:
                        continue

                    # Skip cards already scrolled past
                    if (box["y"] + box["height"]) < 60:
                        continue

                    # Skip cards too far below
                    if box["y"] > (viewport_h + 300):
                        continue

                    # Read text
                    msg_loc = card.locator('div[data-ad-preview="message"], div[data-ad-rendering-role="story_message"]')
                    post_text = ""
                    if msg_loc.count() > 0:
                        post_text = clean_text(msg_loc.first.inner_text(timeout=500))
                    else:
                        colored_loc = card.locator('div[style*="font-size: 30px"], div[style*="font-size: 24px"]')
                        if colored_loc.count() > 0:
                            post_text = clean_text(colored_loc.first.inner_text(timeout=300))

                    if not matches_keywords(post_text):
                        continue

                    # Get permalink
                    url = get_post_permalink(card)
                    sig = url or post_text[:100]

                    if sig in collected_posts:
                        continue

                    # Expand "See more" if needed
                    try:
                        see_more = card.locator('div[role="button"]:has-text("See more"), div[role="button"]:has-text("আরও দেখুন")')
                        if see_more.count() > 0 and see_more.first.is_visible():
                            see_more.first.click(timeout=400, force=True)
                            post_text = clean_text(msg_loc.first.inner_text(timeout=500))
                    except Exception:
                        pass

                    collected_posts[sig] = {
                        "date_time": "",
                        "text": post_text,
                        "url": url,
                    }

                    matched = [w for w in KEYWORDS if w in post_text]
                    print(f"  [Found Post] Matched: {matched} | URL: {url or 'N/A'}")

                except Exception:
                    continue

            print(f"Scroll {scroll_no + 1}/{MAX_SCROLLS} | Unique Matched: {len(collected_posts)}")
            page.evaluate("window.scrollBy(0, Math.floor(window.innerHeight * 0.85))")
            page.wait_for_timeout(SCROLL_WAIT_MS)

        # -------------------------------------------------------------
        # PASS 2: Extract Exact Timestamps using Direct-Link Hover
        # -------------------------------------------------------------
        print(f"\n[+] Pass 1 complete. Extracting exact timestamps for {len(collected_posts)} matched posts...")
        for sig, post in collected_posts.items():
            if post["url"]:
                print(f"  Fetching date for: {post['url']}")
                exact_ts = scrape_post_timestamp_from_url(context, post["url"])
                post["date_time"] = exact_ts
                print(f"  -> Date: {exact_ts or 'Not available'}")

        # Export with Pandas
        df = pd.DataFrame(list(collected_posts.values()))

        json_path = OUTPUT_DIR / "posts.json"
        csv_path = OUTPUT_DIR / "posts.csv"

        df.to_json(json_path, orient="records", force_ascii=False, indent=2)
        df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        print(f"\nDone! Exported {len(df)} posts.")
        print(f"CSV saved to:  {csv_path.resolve()}")
        print(f"JSON saved to: {json_path.resolve()}")

        context.close()


if __name__ == "__main__":
    main()