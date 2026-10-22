import datetime
import json
import os
import re
import socket
import sys
import time
from collections import defaultdict
import unicodedata
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from typing import Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = os.getenv("PAGE_URL", "https://www.facebook.com/KalniBR")
OUTPUT_DIR = Path("output")
POSTS_JSON_PATH = OUTPUT_DIR / "posts.json"
POSTS_CSV_PATH = OUTPUT_DIR / "posts.csv"
BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# =========================================================================
# JAVASCRIPT HELPERS (Reused from Scraper)
# =========================================================================
HELPERS_JS = r"""
const SEE_MORE = /^\s*(…|\.\.\.)?\s*(See more|আরও দেখুন)\s*$/i;
const MSG_SEL = '[data-ad-rendering-role="story_message"], [data-ad-comet-preview="message"], [data-ad-preview="message"]';
const SKIP_SEL = 'a, [role="button"], [role="tooltip"], [role="textbox"], [contenteditable="true"], h2, h3, h4, ' +
                 '[data-ad-rendering-role="profile_name"], [aria-label^="Comment"], [aria-label^="Reply"], ' +
                 'form, ul, [role="toolbar"]';

const serialize = (root) => {
  let s = '';
  const nl = () => { if (s && !s.endsWith('\n')) s += '\n'; };
  const walk = (n) => {
    if (n.nodeType === 3) { s += n.nodeValue; return; }
    if (n.nodeType !== 1) return;
    const t = n.tagName.toLowerCase();
    if (t === 'br') { s += '\n'; return; }
    if (t === 'img') { s += n.alt || ''; return; }
    if (t === 'script' || t === 'style' || t === 'svg') return;
    const role = n.getAttribute('role');
    if (role === 'button' || role === 'tooltip') return;
    const block = (t === 'div' || t === 'p' || t === 'li');
    if (block) nl();
    for (const c of n.childNodes) walk(c);
    if (block) nl();
  };
  walk(root);
  return s.replace(/\u00a0/g, ' ')
          .replace(/[ \t]+\n/g, '\n').replace(/\n[ \t]+/g, '\n')
          .replace(/\n{3,}/g, '\n\n').replace(/…\s*$/, '').trim();
};

const collapse = (s) => {
  const L = s.split('\n'), n = L.length;
  for (let k = 1; k <= n / 2; k++) {
    if (n % k) continue;
    let ok = true;
    for (let i = k; i < n && ok; i++) if (L[i].trim() !== L[i - k].trim()) ok = false;
    if (ok) return L.slice(0, k).join('\n');
  }
  return s;
};

const findRoots = (card) => {
  let roots = [...card.querySelectorAll(MSG_SEL)];
  roots = roots.filter(r => !roots.some(o => o !== r && o.contains(r)));
  if (roots.length) return roots;
  const cands = [...card.querySelectorAll('div')].filter(d =>
    !d.querySelector('div') && !d.closest(SKIP_SEL) && (d.innerText || '').trim().length >= 3);
  if (!cands.length) return [];
  const withBr = cands.filter(d => d.querySelector('br'));
  const pool = withBr.length ? withBr : cands;
  pool.sort((a, b) => b.innerText.length - a.innerText.length);
  return [pool[0]];
};

const markTs = (card, roots) => {
  card.querySelectorAll('[data-ts-link]').forEach(a => a.removeAttribute('data-ts-link'));
  const first = roots[0];
  const PERMA = /\/(posts|videos|reel|reels|permalink|watch|share)\b|story_fbid|fbid=|[?&]v=/;
  const AGE = /^\s*[\d০-৯]+\s*(s|m|h|d|w|y|সে|মি|ঘ|দি|স|ব)/i;
  const collect = (strict) => {
    const cands = [];
    for (const a of card.querySelectorAll('a[href]')) {
      const h = a.getAttribute('href') || '';
      if (/hashtag|\/browse\//.test(h)) continue;
      if (a.closest('[role="button"], [data-ad-rendering-role="profile_name"]')) continue;
      if (roots.some(r => r.contains(a) || a.contains(r))) continue;
      if (strict && first && !(first.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_PRECEDING)) continue;
      let rank = 9;
      if (a.querySelector('[aria-labelledby]')) rank = 0;
      else if (a.querySelector('svg use')) rank = 1;
      else if (h === '#' || h[0] === '?') rank = 2;
      else if (PERMA.test(h)) rank = 3;
      else if (!strict && AGE.test(a.innerText || '')) rank = 4;
      if (rank < 9) cands.push({ a, rank });
    }
    return cands.sort((x, y) => x.rank - y.rank);
  };
  let cands = collect(true);
  if (!cands.length) cands = collect(false);
  cands.slice(0, 3).forEach(c => c.a.setAttribute('data-ts-link', '1'));
  return cands.length > 0;
};

const permalink = (card) => {
  for (const a of card.querySelectorAll('a[href]')) {
    const h = a.getAttribute('href');
    if (!h || h === '#') continue;
    let u; try { u = new URL(h, location.href); } catch (e) { continue; }
    const p = u.pathname;
    if (/\/(hashtag|hashtags|browse)\//.test(p) || p.includes('/watch/hashtag')) continue;
    const ok = /\/(posts|videos|reel|reels|permalink|share)\//.test(p) || p.includes('/photo') ||
               u.searchParams.has('story_fbid') || u.searchParams.has('fbid') ||
               (p.startsWith('/watch') && u.searchParams.has('v'));
    if (!ok) continue;
    for (const k of [...u.searchParams.keys()]) if (k.startsWith('__') || k === 'comment_id') u.searchParams.delete(k);
    return u.origin + p + u.search;
  }
  return '';
};

const extract = (card, onlyFirst = false) => {
  let roots = findRoots(card);
  if (onlyFirst) roots = roots.slice(0, 1);
  const truncated = roots.some(r => [...r.querySelectorAll('[role="button"]')]
                      .some(b => SEE_MORE.test((b.innerText || '').trim())));
  const parts = [...new Set(roots.map(serialize).filter(Boolean))];
  const text = collapse(parts.join('\n'));
  return { text, url: permalink(card), truncated, hasTs: markTs(card, roots) };
};

const getCards = () => {
  let c = [...document.querySelectorAll('div[role="feed"] > div')];
  if (!c.length) c = [...document.querySelectorAll('[aria-posinset]')];
  if (!c.length) c = [...document.querySelectorAll('div[role="article"]')].filter(a => !a.getAttribute('aria-label'));
  return c;
};

const clickSeeMore = (scope) => {
  let n = 0;
  for (const b of scope.querySelectorAll('[role="button"]')) {
    if (SEE_MORE.test((b.innerText || '').trim())) { b.click(); n++; }
  }
  return n;
};
"""

CLICK_FEED_JS = "() => {" + HELPERS_JS + "return clickSeeMore(document);}"
FEED_JS = "() => {" + HELPERS_JS + r"""
  window.__repairSeq = window.__repairSeq || 0;
  const out = [];
  for (const card of getCards()) {
    if (!card.dataset.cardId) card.dataset.cardId = String(++window.__repairSeq);
    const r = extract(card);
    if (r.text || r.url) out.push({ ...r, id: card.dataset.cardId });
  }
  return out;
}"""

MONTH_NAME_MAP = {
    "1": ["January", "Jan", "জানুয়ারি", "জানুয়ারি"],
    "2": ["February", "Feb", "ফেব্রুয়ারি", "ফেব্রুয়ারি"],
    "3": ["March", "Mar", "মার্চ"],
    "4": ["April", "Apr", "এপ্রিল"],
    "5": ["May", "মে"],
    "6": ["June", "Jun", "জুন"],
    "7": ["July", "Jul", "জুলাই"],
    "8": ["August", "Aug", "আগস্ট"],
    "9": ["September", "Sep", "সেপ্টেম্বর"],
    "10": ["October", "Oct", "অক্টোবর"],
    "11": ["November", "Nov", "নভেম্বর"],
    "12": ["December", "Dec", "ডিসেম্বর"],
}


# =========================================================================
# STRING & DATE UTILITIES
# =========================================================================
def is_port_listening(port=9222, host="127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?im)^\s*(…|\.\.\.)?\s*(See more|See less|আরও দেখুন|কম দেখুন)\s*$", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def normalize_match_text(text: str) -> str:
    """Normalizes text for fuzzy matching between posts.json and live cards."""
    if not text:
        return ""
    t = unicodedata.normalize("NFC", text).lower()
    t = re.sub(r"[\s#_\-\u200c\u200d।.,!?:;\"'()]+", " ", t).strip()
    return t


def text_matches(target_text: str, candidate_text: str) -> bool:
    """Evaluates if the live card text corresponds to the missing post."""
    norm_target = normalize_match_text(target_text)
    norm_cand = normalize_match_text(candidate_text)

    if not norm_target or not norm_cand:
        return False

    # 1. Direct Substring Match (First 50 chars)
    anchor_len = min(len(norm_target), 50)
    if anchor_len >= 15 and norm_target[:anchor_len] in norm_cand:
        return True
    if len(norm_cand) >= 15 and norm_cand[:15] in norm_target:
        return True

    # 2. Significant Word Overlap (Jaccard-like)
    target_words = set(norm_target.split())
    cand_words = set(norm_cand.split())
    overlap = len(target_words & cand_words)
    if overlap >= 5 and (overlap / len(target_words)) >= 0.65:
        return True

    return False


def parse_post_dt(s: str) -> Optional[datetime.datetime]:
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})", s)
    if m:
        d, mon, y = m.groups()
        try:
            mon_idx = datetime.datetime.strptime(mon[:3], "%b").month
            return datetime.datetime(int(y), mon_idx, int(d))
        except Exception:
            pass
    return None


def clean_facebook_url(url: str) -> str:
    p = urlparse(url)
    params = [f"{k}={v}" for k, vals in parse_qs(p.query).items()
              if not k.startswith("__") and k != "comment_id" for v in vals]
    return f"{p.scheme}://{p.netloc}{p.path}" + (f"?{'&'.join(params)}" if params else "")


def is_post_url(url: str) -> bool:
    p = urlparse(url)
    if any(b in p.path for b in ("/hashtag", "/browse/", "/watch/hashtag")):
        return False
    return (any(m in p.path for m in ("/posts/", "/videos/", "/reel/", "/reels/", "/photo", "/permalink", "/share/"))
            or any(k in p.query for k in ("story_fbid", "fbid", "v=")))


def is_valid_timestamp_tooltip(text: str) -> bool:
    return bool(text) and len(text) >= 8 and ":" in text and bool(re.search(r"[\d০-৯]", text))


# =========================================================================
# FACEBOOK UI AUTOMATION (FILTERS, SHARE BUTTON & HOVER)
# =========================================================================
def select_dropdown_item(page, target_pattern: str) -> bool:
    pattern = re.compile(rf"^\s*(?:{target_pattern})\s*$", re.I)
    selectors = ['[role="option"]', '[role="menuitemradio"]', '[role="listbox"] [role="button"]', '[role="listbox"] div']
    for sel in selectors:
        items = page.locator(sel).filter(has_text=pattern)
        if items.count() > 0:
            for k in range(items.count()):
                cand = items.nth(k)
                if cand.is_visible():
                    cand.scroll_into_view_if_needed(timeout=1000)
                    cand.click(force=True, timeout=1500)
                    page.wait_for_timeout(600)
                    return True
    return False


def get_day_regex_pattern(day_val) -> str:
    """Matches English digits, a leading zero, and Bengali numerals."""
    d_int = int(re.search(r"\d+", str(day_val).translate(BN_TO_EN)).group(0))
    d_bn = str(d_int).translate(str.maketrans("0123456789", "০১২৩৪৫৬৭৮৯"))
    return rf"0?{d_int}|{d_bn}"

def find_combobox(page, modal, label_pattern: str, fallback_index: int, timeout_ms: int = 8000):
    """Find a filter combobox by its exact label (e.g. 'Day'); fall back to its position in the dialog.
    Waits for the dialog to finish rendering, and raises a clear error if nothing is found."""
    label_re = re.compile(rf"^\s*(?:{label_pattern})\s*$", re.I)
    waited = 0
    while waited <= timeout_ms:
        combos = modal.locator('[role="combobox"]')
        n = combos.count()
        if n:
            labelled = combos.filter(has=page.locator("span", has_text=label_re))
            if labelled.count() > 0:
                return labelled.first
            if n > fallback_index:
                return combos.nth(fallback_index)
        page.wait_for_timeout(250)
        waited += 250

    try:
        snippet = modal.inner_text(timeout=1000)[:150].replace("\n", " | ")
    except Exception:
        snippet = "?"
    raise RuntimeError(
        f"combobox '{label_pattern}' not found "
        f"(in dialog: {modal.locator('[role=combobox]').count()}, on page: {page.locator('[role=combobox]').count()}; "
        f"dialog text: {snippet})"
    )

def apply_date_filter(page, year: str, month: str, day: Optional[str | int] = None):
    """
    Opens Facebook 'Post filters' dialog and sequentially selects:
    1. Year
    2. Month (dynamically injected after Year)
    3. Day   (dynamically injected after Month)
    Then clicks 'Done'.
    """
    if not year and not month and not day:
        return

    day_str = str(day) if day else "Any"
    print(f"\n[+] Applying Date Filter: Year={year or 'Any'}, Month={month or 'Any'}, Day={day_str}...", flush=True)

    # 1. Open 'Post filters' modal
    modal = page.locator('div[role="dialog"]:has-text("Post filters")')
    if not modal.is_visible():
        try:
            filters_btn = page.locator(
                'div[role="button"]:has-text("Filters"), div[aria-label="Filters"], span:has-text("Filters")'
            ).first
            filters_btn.scroll_into_view_if_needed(timeout=2000)
            filters_btn.click(timeout=2500)
            modal.wait_for(state="visible", timeout=6000)
        except Exception as e:
            print(f"    [!] Could not open 'Post filters' modal: {e}", flush=True)
            return

    # 2. Select Year (1st Combobox)
    if year:
        try:
            year_combo = find_combobox(page, modal, "Year|বছর", 0)
            year_combo.scroll_into_view_if_needed(timeout=1000)
            print(f"    [→] Clicking combobox: '{year_combo.inner_text().strip()}'", flush=True)
            year_combo.click(timeout=2000)
            page.wait_for_timeout(600)
            if select_dropdown_item(page, str(year).strip()):
                print(f"    [✔] Year selected: {year}", flush=True)
            else:
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"    [!] Year selection error: {e}", flush=True)

    # 3. Select Month (2nd Combobox - appears after Year)
    if month:
        try:
            month_combo = find_combobox(page, modal, "Month|মাস", 1)
            month_combo.wait_for(state="visible", timeout=4000)
            print(f"    [→] Clicking combobox: '{month_combo.inner_text().strip()}'", flush=True)
            month_combo.click(timeout=2000)
            page.wait_for_timeout(600)

            month_str = str(month).strip()
            match_variants = [month_str]
            for num_k, names in MONTH_NAME_MAP.items():
                if month_str.lower() in [n.lower() for n in names] or month_str == num_k:
                    match_variants.extend(names)
                    break

            combined_month_pattern = "|".join(re.escape(v) for v in set(match_variants))
            if select_dropdown_item(page, combined_month_pattern):
                print(f"    [✔] Month selected: {month}", flush=True)
            else:
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"    [!] Month selection error: {e}", flush=True)

    # 4. Select Day (3rd Combobox - appears after Month)
    if day:
        try:
            day_combo = find_combobox(page, modal, "Day|দিন|তারিখ", 2)
            day_combo.wait_for(state="visible", timeout=4000)
            print(f"    [→] Clicking combobox: '{day_combo.inner_text().strip()}'", flush=True)
            day_combo.click(timeout=2000)
            page.wait_for_timeout(600)

            day_pattern = get_day_regex_pattern(day)
            if select_dropdown_item(page, day_pattern):
                print(f"    [✔] Day selected: {day}", flush=True)
            else:
                print(f"    [!] Could not find Day '{day}' in listbox", flush=True)
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"    [!] Day selection error: {e}", flush=True)

    # 5. Click Done
    try:
        done_btn = modal.locator('div[role="button"]:has-text("Done"), div[aria-label="Done"], div[role="button"]:has-text("সম্পন্ন")').first
        if done_btn.is_visible():
            done_btn.click(timeout=2000)
            modal.wait_for(state="hidden", timeout=5000)
            print("    [✔] Clicked 'Done'. Feed updated to target day.", flush=True)
    except Exception as e:
        print(f"    [!] Could not click 'Done': {e}", flush=True)

    page.wait_for_timeout(3500)


def get_url_via_share_button(page, card_scope: str) -> str:
    """Clicks Share -> Copy link and intercepts the URL."""
    try:
        page.evaluate("""() => {
            window.__lastCopiedUrl = "";
            if (!window.__clipHooked) {
                window.__clipHooked = true;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    const orig = navigator.clipboard.writeText.bind(navigator.clipboard);
                    navigator.clipboard.writeText = async (txt) => {
                        window.__lastCopiedUrl = txt;
                        return orig(txt);
                    };
                }
            }
        }""")

        card = page.locator(card_scope).first
        if not card.is_visible(timeout=1000):
            return ""

        share_selectors = [
            'div[role="button"]:has(i[style*="-567px"])',
            'div[role="button"]:has(i[style*="-150px"])',
            'div[aria-label*="Share" i]',
            'div[aria-label*="শেয়ার" i]',
            'div[aria-label*="শেয়ার" i]',
            'div[role="button"]:has-text("Share")',
            'div[role="button"]:has-text("শেয়ার")',
            'div[role="button"]:has-text("শেয়ার")',
        ]

        share_btn = None
        for sel in share_selectors:
            loc = card.locator(sel)
            if loc.count() > 0 and loc.first.is_visible(timeout=300):
                share_btn = loc.first
                break

        if not share_btn:
            return ""

        share_btn.scroll_into_view_if_needed(timeout=1000)
        share_btn.click(timeout=1500)
        page.wait_for_timeout(600)

        copy_selectors = [
            'div[role="menuitem"]:has-text("Copy link")',
            'div[role="menuitem"]:has-text("কপি লিঙ্ক")',
            'div[role="menuitem"]:has-text("লিংক কপি")',
            'div[role="menuitem"]:has-text("লিঙ্ক কপি")',
            'div[role="button"]:has-text("Copy link")',
            'span:has-text("Copy link")',
            'span:has-text("কপি লিঙ্ক")',
            'span:has-text("লিঙ্ক কপি করুন")',
            'span:has-text("লিংক কপি করুন")',
            '[aria-label*="Copy link" i]',
        ]

        copied_url = ""
        for c_sel in copy_selectors:
            c_btn = page.locator(c_sel).first
            if c_btn.is_visible(timeout=800):
                c_btn.click(timeout=1000)
                page.wait_for_timeout(400)
                copied_url = page.evaluate("() => window.__lastCopiedUrl || ''")
                break

        if not copied_url:
            try:
                copied_url = page.evaluate("() => navigator.clipboard ? navigator.clipboard.readText() : ''")
            except Exception:
                pass

        page.keyboard.press("Escape")
        page.wait_for_timeout(250)

        if copied_url and ("facebook.com" in copied_url or "fb.watch" in copied_url):
            return clean_facebook_url(copied_url)

    except Exception:
        try:
            page.keyboard.press("Escape")
        except Exception:
            pass

    return ""


def hover_timestamp(pg, scope: str = "") -> tuple[str, str]:
    exact_date, url = "", ""
    cands = pg.locator(f"{scope} [data-ts-link]".strip())
    n = min(cands.count(), 3)
    if n == 0:
        return "", ""

    for i in range(n):
        loc = cands.nth(i)
        try:
            loc.evaluate("el => el.scrollIntoView({block: 'center', inline: 'center'})")
            pg.wait_for_timeout(250)

            box = loc.bounding_box()
            if box:
                cx, cy = box["x"] + min(box["width"] / 2, 40), box["y"] + box["height"] / 2
                pg.mouse.move(cx - 8, cy, steps=4)
                pg.mouse.move(cx, cy, steps=4)
            else:
                loc.hover(force=True, timeout=1500)

            # Read tooltip
            for _ in range(4):
                pg.wait_for_timeout(300)
                tips = pg.locator('div[role="tooltip"]')
                for k in range(tips.count()):
                    t = tips.nth(k).inner_text(timeout=400).strip()
                    if is_valid_timestamp_tooltip(t):
                        exact_date = t
                        break
                if exact_date:
                    break

            href = loc.get_attribute("href", timeout=1000) or ""
            if href and href != "#":
                u = clean_facebook_url(urljoin("https://www.facebook.com", href))
                if is_post_url(u):
                    url = u
            if exact_date:
                break
        except Exception:
            pass

    return exact_date, url


def date_from_page_source(html: str, post_url: str = "") -> str:
    times = [(m.start(), int(m.group(1))) for m in
             re.finditer(r'"(?:creation_time|publish_time|created_time|story_publish_time|timestamp)"\s*:\s*(\d{10})\b', html)]
    if not times:
        abbr_times = [int(m.group(1)) for m in re.finditer(r'data-utime=["\'](\d{10})["\']', html)]
        if abbr_times:
            return datetime.datetime.fromtimestamp(abbr_times[0]).strftime("%A %d %B %Y at %H:%M")
        return ""
    chosen = times[0][1]
    return datetime.datetime.fromtimestamp(chosen).strftime("%A %d %B %Y at %H:%M")


def scrape_post_details_from_url(context, post_url: str, main_page) -> tuple[str, str]:
    post_page = None
    exact_date, full_text = "", ""
    try:
        post_page = context.new_page()
        post_page.goto(post_url, wait_until="domcontentloaded", timeout=12000)
        post_page.wait_for_timeout(800)
        exact_date, _ = hover_timestamp(post_page)
        if not exact_date:
            exact_date = date_from_page_source(post_page.content(), post_url)
    except Exception as e:
        print(f"      [!] Tab fetch error: {e}", flush=True)
    finally:
        if post_page:
            try:
                post_page.close()
            except Exception:
                pass
        try:
            main_page.bring_to_front()
        except Exception:
            pass
    return exact_date, full_text


# =========================================================================
# REPAIR ENGINE
# =========================================================================
def find_approx_date_for_item(posts: list, idx: int) -> Tuple[str, str, Optional[int]]:
    """Returns (Year, Month, Day) from the nearest neighbor post with a valid date."""
    # 1. Search backward
    for i in range(idx - 1, -1, -1):
        dt = parse_post_dt(posts[i].get("date_time", ""))
        if dt:
            return str(dt.year), dt.strftime("%B"), dt.day

    # 2. Search forward
    for i in range(idx + 1, len(posts)):
        dt = parse_post_dt(posts[i].get("date_time", ""))
        if dt:
            return str(dt.year), dt.strftime("%B"), dt.day

    now = datetime.datetime.now()
    return str(now.year), now.strftime("%B"), None


def save_repaired_dataset(posts: list):
    try:
        with open(POSTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        df = pd.DataFrame(posts)
        df.to_csv(POSTS_CSV_PATH, index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[!] Error saving repaired posts: {e}", flush=True)


def main():
    if not POSTS_JSON_PATH.exists():
        print(f"[-] File not found: {POSTS_JSON_PATH}")
        return

    with open(POSTS_JSON_PATH, "r", encoding="utf-8") as f:
        posts = json.load(f)

    # 1. Identify posts with missing date_time OR missing url
    incomplete_indices = [
        i for i, p in enumerate(posts)
        if not p.get("url") or not p.get("date_time")
    ]

    if not incomplete_indices:
        print("[🎉 SUCCESS] All posts in posts.json already have complete URLs and date_times!")
        return

    print(f"\n[+] Found {len(incomplete_indices)} posts with missing URL or date_time out of {len(posts)} total.")

# Group incomplete posts into (Year, Month, Day) buckets
    buckets = defaultdict(list)
    for idx in incomplete_indices:
        item = posts[idx]
        y, m, d = find_approx_date_for_item(posts, idx)
        buckets[(y, m, d)].append((idx, item))

    print(f"[+] Partitioned missing posts into {len(buckets)} specific day bucket(s):")
    for (y, m, d), items in buckets.items():
        print(f"    - {y} {m} {d or 'Any'}: {len(items)} post(s)")

    # 3. Connect to Brave Browser
    snap_brave_bin = "/snap/bin/brave"
    snap_profile_dir = os.path.expanduser("~/snap/brave/current/.config/BraveSoftware/Brave-Browser")
    if not os.path.exists(snap_profile_dir):
        snap_profile_dir = os.path.expanduser("~/snap/brave/692/.config/BraveSoftware/Brave-Browser")

    with sync_playwright() as p:
        browser = None
        if is_port_listening(9222):
            print("\n[+] Connecting to Snap Brave on port 9222 via CDP...", flush=True)
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()
        else:
            print(f"\n[*] Launching Snap Brave directly: {snap_profile_dir}", flush=True)
            try:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=snap_profile_dir, executable_path=snap_brave_bin,
                    headless=False, no_viewport=True,
                )
                page = context.pages[0] if context.pages else context.new_page()
            except Exception as e:
                print(f"[!] Please ensure Brave is running: /snap/bin/brave --remote-debugging-port=9222 & ({e})")
                return

        print(f"[+] Navigating to: {PAGE_URL}", flush=True)
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(3000)

        total_repaired = 0

        # 4. Process each bucket
        for (year, month, day), pending_items in buckets.items():
            print(f"\n" + "=" * 70)
            print(f"🔍 JUMPING TO: {year} {month} {day or 'Any'} ({len(pending_items)} targets)")
            print("=" * 70)
        
            # Apply Year, Month AND Day filter
            apply_date_filter(page, year, month, day)

            # Scroll and search in this bucket
            remaining = {idx: item for idx, item in pending_items}
            last_scroll_y = -1
            stalled = 0

            for scroll_no in range(40):  # Up to 40 scrolls per month bucket
                if not remaining:
                    print(f"    [✔] All missing posts for {year} {month} found and repaired!")
                    break

                try:
                    page.evaluate(CLICK_FEED_JS)
                    page.wait_for_timeout(400)
                    cards = page.evaluate(FEED_JS)
                except Exception:
                    cards = []

                for card_info in cards:
                    cid = card_info["id"]
                    card_txt = clean_text(card_info.get("text", ""))
                    card_url = card_info.get("url", "")
                    card_scope = f'[data-card-id="{cid}"]'

                    # Check against all remaining incomplete posts in this bucket
                    matched_idx = None
                    for orig_idx, orig_item in list(remaining.items()):
                        if text_matches(orig_item["text"], card_txt):
                            matched_idx = orig_idx
                            break

                    if matched_idx is None:
                        continue

                    # --- MATCH FOUND: REPAIR URL AND DATE_TIME ---
                    target_post = posts[matched_idx]
                    print(f"\n  [🎯 MATCH FOUND] Repairing post #{matched_idx}...", flush=True)
                    print(f"  Target Text: {target_post['text'][:90]}...", flush=True)

                    # 1. Resolve URL if missing
                    repaired_url = target_post.get("url", "") or card_url
                    if not repaired_url:
                        # Try hover timestamp link first
                        _, u = hover_timestamp(page, card_scope)
                        repaired_url = u

                    if not repaired_url:
                        # Fallback to Share -> Copy link
                        print("    [!] Fetching URL via Share -> Copy link button...", flush=True)
                        repaired_url = get_url_via_share_button(page, card_scope)

                    # 2. Resolve Date if missing
                    repaired_dt = target_post.get("date_time", "")
                    if not repaired_dt:
                        d, _ = hover_timestamp(page, card_scope)
                        repaired_dt = d

                    if not repaired_dt and repaired_url:
                        d, _ = scrape_post_details_from_url(context, repaired_url, page)
                        repaired_dt = d

                    # Update and Save
                    if repaired_url or repaired_dt:
                        target_post["url"] = repaired_url or target_post.get("url", "")
                        target_post["date_time"] = repaired_dt or target_post.get("date_time", "")
                        save_repaired_dataset(posts)
                        total_repaired += 1
                        print(f"    [✔] REPAIRED! Date: {target_post['date_time'] or 'None'} | URL: {target_post['url'] or 'None'}", flush=True)

                    # Remove from remaining list for this bucket
                    del remaining[matched_idx]

                # Scroll down in current month
                y = page.evaluate("() => window.scrollY")
                if y == last_scroll_y:
                    stalled += 1
                    page.keyboard.press("Escape")
                    if stalled >= 2:
                        page.evaluate("window.scrollBy(0, -500)")
                        page.wait_for_timeout(400)
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(800)
                    if stalled >= 10:
                        print(f"    [!] Reached end of feed for {year} {month}.", flush=True)
                        break
                else:
                    stalled, last_scroll_y = 0, y

                page.mouse.wheel(0, 900)
                page.wait_for_timeout(2000)

        if browser:
            browser.close()
        else:
            context.close()

        print(f"\n" + "=" * 70)
        print(f"🎉 FINISHED! Total posts successfully repaired: {total_repaired}")
        print(f"Saved updated dataset to {POSTS_JSON_PATH} and {POSTS_CSV_PATH}")
        print("=" * 70)


if __name__ == "__main__":
    main()