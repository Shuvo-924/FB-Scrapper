import argparse
import datetime
import json
import os
import re
import socket
import sys
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse
from typing import Optional, Tuple

import pandas as pd
from playwright.sync_api import sync_playwright

PAGE_URL = os.getenv("PAGE_URL", "https://www.facebook.com/KalniBR")
MAX_SCROLLS = int(os.getenv("MAX_SCROLLS", "1500"))
SCROLL_WAIT_MS = int(os.getenv("SCROLL_WAIT_MS", "2500"))
FETCH_DATES = os.getenv("FETCH_DATES", "1") == "1"
STALL_LIMIT = int(os.getenv("STALL_LIMIT", "15"))

# Date Filter Settings (Configurable via CLI arguments or environment variables)
FILTER_YEAR = os.getenv("FILTER_YEAR", "2026")        # e.g. "2026", "2025"
FILTER_MONTH = os.getenv("FILTER_MONTH", "August")    # e.g. "August", "8", "আগস্ট"

KEYWORDS_ENV = os.getenv(
    "KEYWORDS",
    "ঢাকা,সিলেট,কমলাপুর,শায়েস্তাগঞ্জ,রাজধানী,লন্ডন,কালনী,৭৭৩,৭৭৪,773,774,kalni,london,dhaka,sylhet,"
    "kamlapur,shaistaganj,kamalapur,শ্রীমঙ্গল,কুলাউড়া,মাইজগাঁও,ভৈরব,আজমপুর",
)
KEYWORDS = [k.strip().lower() for k in KEYWORDS_ENV.split(",") if k.strip()]

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
COLLECTED_POSTS = {}

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

MONTH_REGEX_MAP = {
    1: [r"\bjan(?:uary)?\b", r"জানুয়া?রি"],
    2: [r"\bfeb(?:ruary)?\b", r"ফেব্রুয়া?রি"],
    3: [r"\bmar(?:ch)?\b", r"মার্চ"],
    4: [r"\bapr(?:il)?\b", r"এপ্রিল"],
    5: [r"\bmay\b", r"মে"],
    6: [r"\bjun(?:e)?\b", r"জুন"],
    7: [r"\bjul(?:y)?\b", r"জুলাই"],
    8: [r"\baug(?:ust)?\b", r"আগস্ট"],
    9: [r"\bsep(?:tember)?\b", r"সেপ্টেম্বর"],
    10: [r"\boct(?:ober)?\b", r"অক্টোবর"],
    11: [r"\bnov(?:ember)?\b", r"নভেম্বর"],
    12: [r"\bdec(?:ember)?\b", r"ডিসেম্বর"],
}

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

def parse_month_year(date_str: str) -> Tuple[Optional[int], Optional[int]]:
    """Accurately parses (year, month) using word boundaries to prevent '31' matching '1'."""
    if not date_str:
        return None, None
    s = str(date_str).translate(BN_TO_EN)

    # 1. Extract 4-digit Year (2020-2030)
    y_match = re.search(r"\b(202\d)\b", s)
    year = int(y_match.group(1)) if y_match else None

    # 2. Extract Month using strict word boundary regex
    month = None
    s_lower = s.lower()
    for m_num, patterns in MONTH_REGEX_MAP.items():
        for pat in patterns:
            if re.search(pat, s_lower, re.I):
                month = m_num
                break
        if month:
            break

    # 3. Fallback for numeric ISO date formats (e.g. '2026-07-31')
    if not month:
        iso_m = re.search(r"\b202\d-(\d{1,2})-\d{1,2}\b", s)
        if iso_m:
            month = int(iso_m.group(1))

    return year, month


def get_target_year_month(year_str: str, month_str: str) -> Tuple[Optional[int], Optional[int]]:
    y = None
    if year_str:
        m = re.search(r"\b(202\d)\b", str(year_str).translate(BN_TO_EN))
        if m:
            y = int(m.group(1))

    m_num = None
    if month_str:
        m_clean = str(month_str).lower().strip()
        for num_k, patterns in MONTH_REGEX_MAP.items():
            for pat in patterns:
                if re.search(pat, m_clean, re.I):
                    m_num = num_k
                    break
            if m_num:
                break
        if not m_num and m_clean.isdigit():
            m_num = int(m_clean)

    return y, m_num


# =========================================================================
# IN-PAGE JS HELPERS
# =========================================================================
HELPERS_JS = r"""
const SEE_MORE = /^\s*(…|\.\.\.)?\s*(See more|আরও দেখুন)\s*$/i;
const TIME_RE = /[\d০-৯]|just now|yesterday|এইমাত্র|গতকাল/i;
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

const pickRoot = () => {
  const ds = [...document.querySelectorAll('div[role="dialog"]')];
  const good = ds.find(d => d.querySelector(MSG_SEL)) ||
               ds.find(d => d.querySelector('[data-ad-rendering-role="profile_name"], [role="article"]'));
  return good || document.querySelector('div[role="main"]') || document.body;
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
CLICK_POST_JS = "() => {" + HELPERS_JS + "return clickSeeMore(pickRoot());}"
DETAIL_JS = "() => {" + HELPERS_JS + "return extract(pickRoot(), true);}"
FEED_JS = "() => {" + HELPERS_JS + r"""
  window.__seq = window.__seq || 0;
  const out = [];
  for (const card of getCards()) {
    if (card.dataset.scraped) continue;
    if (!card.dataset.cardId) card.dataset.cardId = String(++window.__seq);
    const r = extract(card);
    if (r.text || r.url) out.push({ ...r, id: card.dataset.cardId });
  }
  return out;
}"""
MARK_SCRAPED_JS = """(id) => { const c = document.querySelector('[data-card-id="' + id + '"]'); if (c) c.dataset.scraped = '1'; }"""


# =========================================================================
# FILTERS AUTOMATION (Year & Month Comboboxes)
# =========================================================================
def select_dropdown_item(page, target_pattern: str) -> bool:
    pattern = re.compile(rf"^\s*{target_pattern}\s*$", re.I)
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


def apply_date_filter(page, year: str, month: str):
    if not year and not month:
        return

    print(f"\n[+] Applying Date Filter: Year={year or 'Any'}, Month={month or 'Any'}...", flush=True)

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
            print(f"    [!] Could not open 'Post filters' dialog: {e}", flush=True)
            return

    if year:
        try:
            year_combo = modal.locator('div[role="combobox"]:has-text("Year"), div[role="combobox"]').first
            year_combo.scroll_into_view_if_needed(timeout=1000)
            year_combo.click(timeout=2000)
            page.wait_for_timeout(600)
            if select_dropdown_item(page, str(year).strip()):
                print(f"    [✔] Year selected: {year}", flush=True)
            else:
                page.keyboard.press("Escape")
        except Exception as e:
            print(f"    [!] Error clicking Year combobox: {e}", flush=True)

    if month:
        try:
            month_combo = modal.locator('div[role="combobox"]:has-text("Month")').first
            if not month_combo.is_visible() and modal.locator('div[role="combobox"]').count() >= 2:
                month_combo = modal.locator('div[role="combobox"]').nth(1)

            month_combo.wait_for(state="visible", timeout=4000)
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
            print(f"    [!] Month combobox failed: {e}", flush=True)

    try:
        done_btn = modal.locator('div[role="button"]:has-text("Done"), div[aria-label="Done"], div[role="button"]:has-text("সম্পন্ন")').first
        if done_btn.is_visible():
            done_btn.click(timeout=2000)
            modal.wait_for(state="hidden", timeout=5000)
            print("    [✔] Clicked 'Done'. Feed is updating...", flush=True)
    except Exception as e:
        print(f"    [!] Could not click 'Done': {e}", flush=True)

    page.wait_for_timeout(3500)
    print("    [✔] Date filter applied successfully. Ready to scrape.\n", flush=True)


# =========================================================================
# SHARE BUTTON LINK FETCH FALLBACK
# =========================================================================
def get_url_via_share_button(page, card_scope: str) -> str:
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


# =========================================================================
# Python Helpers
# =========================================================================
def is_port_listening(port=9222, host="127.0.0.1") -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex((host, port)) == 0
    except Exception:
        return False


def load_existing_data():
    f = OUTPUT_DIR / "posts.json"
    if f.exists():
        try:
            with open(f, "r", encoding="utf-8") as fh:
                for item in json.load(fh):
                    COLLECTED_POSTS[item.get("url") or item.get("text", "")[:100]] = item
            print(f"[+] Loaded {len(COLLECTED_POSTS)} existing posts", flush=True)
        except Exception as e:
            print(f"[!] Warning reading existing posts: {e}", flush=True)


def save_data():
    if not COLLECTED_POSTS:
        return
    try:
        df = pd.DataFrame(list(COLLECTED_POSTS.values()))
        df.to_json(OUTPUT_DIR / "posts.json", orient="records", force_ascii=False, indent=2)
        df.to_csv(OUTPUT_DIR / "posts.csv", index=False, encoding="utf-8-sig")
    except Exception as e:
        print(f"[!] Error auto-saving data: {e}", flush=True)


def matches_keywords(text: str) -> bool:
    return bool(text) and any(w in text.lower() for w in KEYWORDS)


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"(?im)^\s*(…|\.\.\.)?\s*(See more|See less|আরও দেখুন|কম দেখুন)\s*$", "", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


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


def _clear_tooltips(pg):
    pg.mouse.move(2, 2)
    for _ in range(8):
        if pg.locator('div[role="tooltip"]').count() == 0:
            return
        pg.wait_for_timeout(200)


def _read_tooltip(pg) -> str:
    for _ in range(5):
        pg.wait_for_timeout(350)
        tips = pg.locator('div[role="tooltip"]')
        for k in range(tips.count()):
            try:
                t = tips.nth(k).inner_text(timeout=500).strip()
            except Exception:
                continue
            if is_valid_timestamp_tooltip(t):
                return t
    return ""


def date_from_page_source(html: str, post_url: str = "") -> str:
    times = [(m.start(), int(m.group(1))) for m in
             re.finditer(r'"(?:creation_time|publish_time|created_time|story_publish_time|timestamp)"\s*:\s*(\d{10})\b', html)]
    if not times:
        abbr_times = [int(m.group(1)) for m in re.finditer(r'data-utime=["\'](\d{10})["\']', html)]
        if abbr_times:
            return datetime.datetime.fromtimestamp(abbr_times[0]).strftime("%A %d %B %Y at %H:%M")
        return ""
    chosen = None
    ids = re.findall(r"(\d{12,})", post_url or "")
    for pid in ids:
        pos = [m.start() for m in re.finditer(re.escape(pid), html)]
        if pos:
            best = min(((abs(tp - p), ts) for tp, ts in times for p in pos), key=lambda x: x[0])
            if best[0] < 4000:
                chosen = best[1]
                break
    if chosen is None:
        chosen = times[0][1]
    return datetime.datetime.fromtimestamp(chosen).strftime("%A %d %B %Y at %H:%M")


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
            _clear_tooltips(pg)

            box = loc.bounding_box()
            if box:
                cx, cy = box["x"] + min(box["width"] / 2, 40), box["y"] + box["height"] / 2
                pg.mouse.move(cx - 8, cy, steps=4)
                pg.mouse.move(cx, cy, steps=4)
            else:
                loc.hover(force=True, timeout=1500)
            exact_date = _read_tooltip(pg)

            if not exact_date:
                loc.evaluate("""el => { const r = el.getBoundingClientRect();
                    for (const t of ['mouseover', 'mouseenter', 'mousemove'])
                      el.dispatchEvent(new MouseEvent(t, {bubbles: true, clientX: r.x + 5, clientY: r.y + r.height / 2})); }""")
                exact_date = _read_tooltip(pg)

            if not exact_date:
                try:
                    lab = loc.locator("[aria-labelledby]").first.get_attribute("aria-labelledby", timeout=500)
                    if lab:
                        t = pg.locator(f'[id="{lab}"]').first.inner_text(timeout=500).strip()
                        if is_valid_timestamp_tooltip(t):
                            exact_date = t
                except Exception:
                    pass

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


def scrape_post_details_from_url(context, post_url: str, main_page) -> tuple[str, str]:
    post_page = None
    exact_date, full_text = "", ""
    try:
        post_page = context.new_page()
        post_page.goto(post_url, wait_until="domcontentloaded", timeout=15000)
        try:
            post_page.wait_for_selector('div[role="dialog"], div[role="main"]', timeout=5000)
        except Exception:
            pass
        post_page.wait_for_timeout(800)
        post_page.evaluate(CLICK_POST_JS)
        post_page.wait_for_timeout(300)
        res = post_page.evaluate(DETAIL_JS)
        full_text = clean_text(res.get("text", ""))
        exact_date, _ = hover_timestamp(post_page)
        if not exact_date:
            exact_date = date_from_page_source(post_page.content(), post_url)
    except Exception as e:
        print(f"      [!] Post page fetch error: {e}", flush=True)
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
# MAIN (WITH AUTOMATIC PREVIOUS-MONTH STOP)
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Facebook Page Timeline Scraper with Auto-Stop")
    parser.add_argument("--year", type=str, default=FILTER_YEAR, help="Target year to scrape (e.g. 2026)")
    parser.add_argument("--month", type=str, default=FILTER_MONTH, help="Target month to scrape (e.g. August)")
    args = parser.parse_args()

    filter_year = args.year
    filter_month = args.month

    # Resolve target year and month into numeric integers for boundary checks
    target_y, target_m = get_target_year_month(filter_year, filter_month)
    target_epoch_month = (target_y * 12 + target_m) if (target_y and target_m) else None

    if target_epoch_month:
        print(f"[+] Active Boundary Gate: Will automatically halt when feed crosses before {filter_month} {filter_year}.", flush=True)

    load_existing_data()

    snap_brave_bin = "/snap/bin/brave"
    snap_profile_dir = os.path.expanduser("~/snap/brave/current/.config/BraveSoftware/Brave-Browser")
    if not os.path.exists(snap_profile_dir):
        snap_profile_dir = os.path.expanduser("~/snap/brave/692/.config/BraveSoftware/Brave-Browser")

    with sync_playwright() as p:
        browser = None
        if is_port_listening(9222):
            print("\n[+] Connecting to Brave over CDP...", flush=True)
            browser = p.chromium.connect_over_cdp("http://127.0.0.1:9222")
            context = browser.contexts[0]
            page = context.pages[0] if context.pages else context.new_page()
        else:
            print(f"\n[*] Launching Brave with profile: {snap_profile_dir}", flush=True)
            try:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=snap_profile_dir, executable_path=snap_brave_bin,
                    headless=False, no_viewport=True,
                )
                page = context.pages[0] if context.pages else context.new_page()
            except Exception as e:
                if "SingletonLock" in str(e) or "already in use" in str(e):
                    print("\n⚠️  BRAVE IS ALREADY RUNNING. Start it with:")
                    print("   /snap/bin/brave --remote-debugging-port=9222 > /dev/null 2>&1 &")
                    print("then run this script again.\n")
                    return
                raise

        print(f"[+] Navigating to: {PAGE_URL}", flush=True)
        page.goto(PAGE_URL, wait_until="domcontentloaded", timeout=45000)
        page.wait_for_timeout(3000)

        # Apply Year & Month filters
        if filter_year or filter_month:
            apply_date_filter(page, filter_year, filter_month)

        last_scroll_y, stalled = -1, 0
        has_seen_target_month = False
        consecutive_older_posts = 0
        should_stop_scraping = False

        # Scroll & Scrape Loop
        for scroll_no in range(MAX_SCROLLS):
            if should_stop_scraping:
                break

            try:
                page.evaluate(CLICK_FEED_JS)
                page.wait_for_timeout(500)
                items = page.evaluate(FEED_JS)
            except Exception as e:
                print(f"  [!] Feed read error: {e}", flush=True)
                items = []

            for it in items:
                cid, url, text = it["id"], it.get("url", ""), clean_text(it.get("text", ""))
                card_scope = f'[data-card-id="{cid}"]'

                if not text and not url:
                    continue

                def done():
                    page.evaluate(MARK_SCRAPED_JS, cid)

                if url and url in COLLECTED_POSTS:
                    done()
                    continue

                exact_ts = ""

                # Expand truncated text if URL exists
                if it.get("truncated") and url:
                    exact_ts, full_t = scrape_post_details_from_url(context, url, page)
                    if len(full_t) > len(text):
                        text = full_t

                # Date hover on feed card
                if FETCH_DATES or not url:
                    d, u = hover_timestamp(page, card_scope)
                    exact_ts = exact_ts or d
                    if u and (not url or "/photo" in url):
                        url = u

                # Share button fallback if URL is missing
                if not url:
                    copied_u = get_url_via_share_button(page, card_scope)
                    if copied_u:
                        url = copied_u

                # If URL found but date is missing, fetch from dedicated page
                if FETCH_DATES and url and not exact_ts:
                    d, full_t = scrape_post_details_from_url(context, url, page)
                    exact_ts = d
                    if len(full_t) > len(text):
                        text = full_t

                # =========================================================
# =========================================================
                # BOUNDARY GATE: CHECK IF REACHED PREVIOUS MONTH
                # =========================================================
                if exact_ts and target_epoch_month:
                    post_y, post_m = parse_month_year(exact_ts)
                    if post_y and post_m:
                        post_epoch = post_y * 12 + post_m

                        if post_epoch == target_epoch_month:
                            has_seen_target_month = True
                            consecutive_older_posts = 0
                        elif post_epoch < target_epoch_month:
                            consecutive_older_posts += 1
                            # Only stop once we have actually scrolled through July
                            # and hit 2 consecutive posts from June (or earlier)
                            if has_seen_target_month and consecutive_older_posts >= 2:
                                print(f"\n" + "=" * 70)
                                print(f"🛑 [AUTO-STOP TRIGGERED] Reached previous month: {post_y}-{post_m:02d} ({exact_ts})")
                                print(f"   Target {filter_month} {filter_year} complete. Exiting cleanly.")
                                print("=" * 70 + "\n", flush=True)
                                should_stop_scraping = True
                                done()
                                break

                sig = url or (f"{exact_ts}|{text[:100]}" if exact_ts else text[:100])
                if sig in COLLECTED_POSTS:
                    done()
                    continue

                COLLECTED_POSTS[sig] = {"date_time": exact_ts, "text": text, "url": url}
                save_data()
                done()
                print(f"\n  [🎯 Matched #{len(COLLECTED_POSTS)}] {exact_ts or 'No Date'} | {len(text)} chars", flush=True)
                print(f"  {text[:120].replace(chr(10), ' | ')}", flush=True)
                print(f"  URL: {url or 'N/A'}\n", flush=True)

            if should_stop_scraping:
                break

            y = page.evaluate("() => window.scrollY")
            if y == last_scroll_y:
                stalled += 1
                page.keyboard.press("Escape")
                if stalled >= 2:
                    page.evaluate("window.scrollBy(0, -600)")
                    page.wait_for_timeout(500)
                    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    page.wait_for_timeout(1000)
                if stalled >= STALL_LIMIT:
                    print("[!] Feed no longer loading - reached the end.", flush=True)
                    break
            else:
                stalled, last_scroll_y = 0, y

            page.mouse.wheel(0, 900)
            page.wait_for_timeout(SCROLL_WAIT_MS)

            if (scroll_no + 1) % 3 == 0:
                print(f"[Scroll {scroll_no + 1}/{MAX_SCROLLS}] Y={y}px | New cards: {len(items)} | Total: {len(COLLECTED_POSTS)}", flush=True)

        if browser:
            browser.close()
        else:
            context.close()
        print(f"\n🎉 Done. Target month completed. Total posts saved: {len(COLLECTED_POSTS)}", flush=True)


if __name__ == "__main__":
    main()