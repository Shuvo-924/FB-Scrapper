import hashlib
import json
import os
import re
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from statistics import median
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY! Set it in your environment or .env file.")
client = genai.Client(api_key=GEMINI_API_KEY)

MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.1-flash-lite").split(",")]
BATCH_SIZE = 15
SCHEMA_VERSION = "v4-ops"          # bump when the extraction schema/prompt changes -> old cache entries are ignored

ON_TIME_TOL_MIN = 15       # arrival delay above this => is_late = 1
NEAR_STATION_MIN = 200     # a reported station within this many scheduled minutes is "close"
TURNAROUND_MIN = 75        # rake turnaround 774 arrival -> 773 departure at Dhaka (observed ~55-100)
MIN_DELAY_MIN = -30        # earlier than this = misread
MAX_DELAY_MIN = 420        # later than this = misread / wrong train
OUTLIER_MIN = 150          # drop observations this far from the journey's median delay
MIN_FLAG_COUNT = 3         # a train gets its own crossed_/overtaken_by_ column if it appears this often
LOCO_MIN, LOCO_MAX = 2600, 3399   # plausible BR engine numbers; rejects years (2026), times, crowd counts (3000 জন)
STRONG_LOCO_MIN = 3000     # engine number >= 3000 = strong. Lower numbers are NOT "weak": just standard

OUT = Path("output")
OUT.mkdir(exist_ok=True)
CACHE_FILE = OUT / "extraction_cache.json"

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# =========================================================================
# TIMETABLES
# =========================================================================
def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _build(rows):
    return [{"station": s, "arr_m": _hm(a) if a else None, "dep_m": _hm(d) if d else None} for s, a, d in rows]


T773 = _build([
    ("Dhaka", None, "14:55"), ("Biman Bandar", "15:17", "15:22"), ("Narsingdi", "15:55", "15:57"),
    ("Bhairab Bazar", "16:20", "16:22"), ("Brahmanbaria", "16:46", "16:49"), ("Azampur", "17:12", "17:14"),
    ("Harashpur", "17:34", "17:36"), ("Montola", "17:50", "17:52"), ("Shaistaganj", "18:12", "18:15"),
    ("Noapara", "18:25", "18:27"), ("Rashidpur", "18:35", "18:37"), ("Satgaon", "18:45", "18:47"),
    ("Sreemangal", "18:52", "18:55"), ("Bhanugach", "19:10", "19:12"), ("Shamshernagar", "19:22", "19:26"),
    ("Kulaura", "19:49", "19:51"), ("Baramchal", "20:05", "20:07"), ("Maijgaon", "20:20", "20:23"),
    ("Moglabazar", "20:55", "20:57"), ("Sylhet", "21:30", None),
])
T774 = _build([
    ("Sylhet", None, "06:15"), ("Moglabazar", "06:35", "06:37"), ("Maijgaon", "06:55", "06:57"),
    ("Baramchal", "07:10", "07:12"), ("Kulaura", "07:23", "07:26"), ("Shamshernagar", "07:50", "07:52"),
    ("Bhanugach", "08:05", "08:07"), ("Sreemangal", "08:17", "08:20"), ("Satgaon", "08:30", "08:32"),
    ("Rashidpur", "08:42", "08:44"), ("Shaistaganj", "08:57", "09:00"), ("Noapara", "09:12", "09:14"),
    ("Montola", "09:30", "09:32"), ("Harashpur", "09:52", "09:54"), ("Azampur", "10:20", "10:22"),
    ("Brahmanbaria", "10:48", "10:51"), ("Bhairab Bazar", "11:10", "11:13"), ("Narsingdi", "11:43", "11:45"),
    ("Biman Bandar", "12:24", "12:28"), ("Dhaka", "12:55", None),
])
TT = {"773": {e["station"]: e for e in T773}, "774": {e["station"]: e for e in T774}}
ROUTE = {"773": [e["station"] for e in T773], "774": [e["station"] for e in T774]}
DIR_TO_TRAIN = {"dhaka_to_sylhet": "773", "sylhet_to_dhaka": "774"}
# rake turnaround slack (scheduled minutes between arrival of the inbound service and the next departure)
SLACK_773_FROM_774 = (TT["773"]["Dhaka"]["dep_m"] - TT["774"]["Dhaka"]["arr_m"]) % 1440     # same day at Dhaka
SLACK_774_FROM_773 = (TT["774"]["Sylhet"]["dep_m"] - TT["773"]["Sylhet"]["arr_m"]) % 1440   # overnight at Sylhet


def _nm(s):
    s = unicodedata.normalize("NFC", str(s)).lower()
    return re.sub(r"[\s#_\-\u200c\u200d]", "", s)


STATION_ALIASES = {
    "Dhaka": ["dhaka", "ঢাকা", "কমলাপুর", "kamalapur", "kamlapur", "রাজধানী"],
    "Biman Bandar": ["বিমানবন্দ", "biman", "airport", "এয়ারপোর্ট"],
    "Narsingdi": ["narsingdi", "নরসিংদী"],
    "Bhairab Bazar": ["bhairab", "ভৈরব"],
    "Brahmanbaria": ["brahmanbaria", "ব্রাহ্মণবাড়িয়া", "ব্রাহ্মণবাড়ীয়া", "বিবাড়িয়া", "bbaria"],
    "Azampur": ["azampur", "আজমপুর"],
    "Harashpur": ["harashpur", "হরষপুর", "হারাশপুর", "হরশপুর"],
    "Montola": ["montola", "মনতলা", "মোনতলা"],
    "Shaistaganj": ["shaistaganj", "shayestaganj", "শায়েস্তাগঞ্জ", "শায়েস্থাগঞ্জ", "শায়েস্তাগঞ্জ"],
    "Noapara": ["noapara", "নোয়াপাড়া", "নওয়াপাড়া"],
    "Rashidpur": ["rashid", "রশিদ"],
    "Satgaon": ["satgaon", "সাতগাঁও", "সাতগাও"],
    "Sreemangal": ["sreemangal", "srimangal", "শ্রীমঙ্গল"],
    "Bhanugach": ["bhanugach", "ভানুগাছ"],
    "Shamshernagar": ["shamshernagar", "শমশেরনগর", "শমসেরনগর"],
    "Kulaura": ["kulaura", "কুলাউড়া"],
    "Baramchal": ["baramchal", "বরমচাল"],
    "Maijgaon": ["maijgaon", "মাইজগাঁও", "মাইজগাও"],
    "Moglabazar": ["moglabazar", "মোগলাবাজার"],
    "Sylhet": ["sylhet", "সিলেট", "লন্ডন"],
}
_ALIASES = {c: [_nm(a) for a in al] for c, al in STATION_ALIASES.items()}


def canonical_station(raw):
    if not raw:
        return None
    r = _nm(raw)
    for canon, al in _ALIASES.items():
        if any(a in r for a in al):
            return canon
    return None  # stations outside the timetable (Tejgaon, Tongi, ...) carry no schedule


# other trains that cross / overtake Kalni
TRAIN_ALIASES = {
    "Parabat": ["পারাবত", "parabat", "৭১০"],
    "Udayan": ["উদয়ন", "udayan"],
    "Jayantika": ["জয়ন্তিকা", "joyantika", "jayantika", "৭১৮"],
    "Upaban": ["উপবন", "upaban"],
    "Suborno": ["সুবর্ণা", "সুবর্ণ", "suborno"],
    "Upokul": ["উপকূল", "upokul"],
    "Brahmaputra": ["ব্রহ্মপুত্র", "brahmaputra"],
    "Turag": ["তুরাগ", "turag"],
    "Chattala": ["চট্টলা", "chattala"],
}
_TRAIN_ALIASES = {c: [_nm(a) for a in al] for c, al in TRAIN_ALIASES.items()}


def canon_train(raw):
    if not raw:
        return None
    r = _nm(str(raw).translate(BN_TO_EN))
    for canon, al in _TRAIN_ALIASES.items():
        if any(_nm(a.translate(BN_TO_EN)) in r for a in al):
            return canon
    return str(raw).strip() or None


# drivers: strip honorifics so spelling variants merge. Add manual merges to DRIVER_ALIASES.
DRIVER_NOISE = {"ভাই", "নানাভাই", "নানা", "এল", "এম", "এলএম", "lm", "প্রিয়", "এডমিনের", "এডমিন", "চালক",
                "চালকের", "স্যার", "মামা", "আপা", "ভাইয়া", "দাদা", "চাচা"}
DRIVER_ALIASES = {}       # e.g. {"তৌফিক হোসেন": "তৌফিক"}
LOCO_TYPE_MAP = {}        # optional: {"2901": "<class/model>"}; default type = number series (29xx / 30xx)


def canon_driver(raw):
    if not raw:
        return None
    s = unicodedata.normalize("NFC", str(raw)).lower()
    # keep letters, combining marks (Bengali vowel signs / virama) and digits; \w would drop the marks
    s = "".join(ch if (unicodedata.category(ch)[0] in "LMN" or ch.isspace()) else " " for ch in s)
    name = " ".join(w for w in s.split() if w not in DRIVER_NOISE).strip()
    return DRIVER_ALIASES.get(name, name) or None


# =========================================================================
# TIME HANDLING
# =========================================================================
MONTHS = {m: i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], 1)}


def parse_post_dt(s):
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s+at\s+(\d{1,2}):(\d{2})", s or "")
    if not m:
        return None
    d, mon, y, hh, mm = m.groups()
    try:
        return datetime(int(y), MONTHS[mon.capitalize()], int(d), int(hh), int(mm))
    except Exception:
        return None


CLOCK_RE = re.compile(r"(\d{1,2})\s*[:.ঃ]\s*(\d{2})")


def parse_clock(t):
    """'০৪:২০' / '৭.৫৮' / '10ঃ24' / '3:45 PM' -> (hour, minute, explicit_am_pm)"""
    if not t:
        return None
    s = unicodedata.normalize("NFC", str(t)).translate(BN_TO_EN)
    m = CLOCK_RE.search(s)
    if not m:
        return None
    h, mi = int(m.group(1)), int(m.group(2))
    if mi > 59 or h > 24:
        return None
    low = s.lower()
    if "pm" in low and h < 12:
        h += 12
    elif "am" in low and h == 12:
        h = 0
    return h % 24, mi, ("am" in low or "pm" in low)


def resolve_actual_dt(post_dt, h, m, explicit):
    """Posts use a 12h clock with no AM/PM. The event happened closest to the post's own 24h timestamp."""
    if explicit or h >= 13 or h == 0:
        hours = [h]
    elif h == 12:
        hours = [12, 0]
    else:
        hours = [h, h + 12]
    best = None
    for day_off in (-1, 0, 1):
        base = (post_dt + timedelta(days=day_off)).replace(hour=0, minute=0, second=0, microsecond=0)
        for hh in hours:
            cand = base + timedelta(hours=hh, minutes=m)
            diff = abs((cand - post_dt).total_seconds())
            if best is None or diff < best[0]:
                best = (diff, cand)
    return best[1]


def journey_date_for(train, actual):
    # 773 can reach Sylhet after midnight; journey date stays the departure date
    return (actual - timedelta(hours=6)).date() if train == "773" else actual.date()


def day0(d):
    return datetime.combine(d, datetime.min.time())


def t24(dt):
    return dt.strftime("%H:%M") if dt else None


def mm_to_hhmm(m):
    return f"{(m % 1440) // 60:02d}:{(m % 1440) % 60:02d}" if m is not None else None


# =========================================================================
# GEMINI EXTRACTION (per-post cache)
# =========================================================================
class Obs(BaseModel):
    station: str
    event: str                          # arrival | departure | pass | halt | approaching | waiting
    time_text: Optional[str] = None     # clock time copied exactly as written
    is_forecast: Optional[bool] = None


class Crossing(BaseModel):
    other_train: Optional[str] = None
    station: Optional[str] = None
    wait_minutes: Optional[int] = None
    kalni_waited: Optional[bool] = None   # True: Kalni was held for the other train; False: the other train waited


class Overtake(BaseModel):
    other_train: Optional[str] = None     # train dispatched / passed AHEAD of Kalni while Kalni was held
    station: Optional[str] = None
    wait_minutes: Optional[int] = None


class PostExtraction(BaseModel):
    post_index: int
    is_relevant: bool
    train_number: Optional[str] = None
    direction: Optional[str] = None       # dhaka_to_sylhet | sylhet_to_dhaka
    observations: Optional[List[Obs]] = None
    stated_delay_minutes: Optional[int] = None
    stated_delay_reference: Optional[str] = None   # from_origin | en_route
    locomotive_number: Optional[str] = None
    driver_name: Optional[str] = None
    crossings: Optional[List[Crossing]] = None
    overtaken_by: Optional[List[Overtake]] = None
    delay_reasons: Optional[List[str]] = None


REASONS = [
    "late_engine_or_rake",        # engine / rake not ready or connected late at the origin
    "engine_failure",             # breakdown / electrical failure of Kalni's engine
    "crossing_wait",              # Kalni held for a crossing
    "overtaken_by_other_train",   # another train dispatched or passed ahead of Kalni
    "outer_signal_wait",          # held at an outer signal / could not enter a station
    "platform_blocked",           # platform occupied by another train
    "crew_issue",                 # driver or crew unavailable, rest, changeover problem
    "vip_or_special_movement",
    "track_or_signal_fault",
    "other_operational",
]

SYSTEM_PROMPT = f"""
You extract structured railway-operations data from Bengali Facebook posts (formal and informal, slangy,
often sarcastic) about the Kalni Express (Bangladesh Railway, Dhaka-Sylhet line). Return one entry per post.
Delays on this line come from operations (engines, drivers, crossings, other trains being given priority),
NOT from weather: ignore weather completely.

Train 773 = Dhaka -> Sylhet (scheduled dep 14:55, arr 21:30). Train 774 = Sylhet -> Dhaka (dep 06:15, arr 12:55).
Route Dhaka->Sylhet: Dhaka(Kamalapur), Biman Bandar(Airport), Narsingdi, Bhairab Bazar, Brahmanbaria, Azampur,
Harashpur, Montola, Shaistaganj, Noapara, Rashidpur, Satgaon, Sreemangal, Bhanugach, Shamshernagar, Kulaura,
Baramchal, Maijgaon, Moglabazar, Sylhet. Other stations (Tejgaon, Banani, Cantonment, Tongi, Ghorashal...) may
appear: still output them, name as written.

GLOSSARY: লন্ডন/লন্ডনগামী = Sylhet; কমলাপুর/রাজধানী = Dhaka; ইন = arrival; আউট/ছেড়ে গেল/ছাড়ল/ছেড়ে আসলো = departure;
অতিক্রম/পাস/থ্রু = pass; বসা/দাঁড়িয়ে আছে/আউটারে = halt; আসিতেছে/ইন হচ্ছে/প্রবেশ করছে/প্রবেশের অপেক্ষায় = approaching;
ইঞ্জিন বিহীন/ইঞ্জিন কানেক্ট হয়নি/ছাড়তে পারেনি at the origin = waiting; ক্রসিং = crossing with another train;
এলএম/LM/চালক = driver; বিলম্ব/লেট = delay; ঢাকা টু সিলেট / সিলেটগামী / সিলেটের উদ্দেশ্যে = 773;
সিলেট টু ঢাকা / ঢাকা অভিমুখী / ঢাকাগামী = 774.

RULES
1. is_relevant=true only if the post gives the train's time/position, a quantified delay, the engine number, the driver, a crossing, an overtaking/priority event, or a cause of delay. Greetings, photo credits,
   praise, general rants with no facts, or posts about other trains only -> false. A rant that still states a
   time/position of Kalni is relevant: extract only those facts.
2. observations: one per station event. time_text = the clock time copied EXACTLY as written (e.g. "০৪:২০", "৭.৫৮",
   "3:15", "১০ঃ২৪"). Never convert, never guess. If the post gives no clock time for an event, omit that event.
3. is_forecast=true when the time is a future expectation ("ইনশাআল্লাহ ... হব", "ঢুকবে", "ছেড়ে যাওয়ার কথা").
4. train_number / direction ONLY if the text makes it explicit. Otherwise null. Never guess.
5. stated_delay_minutes only if the post quantifies a delay ("এক ঘন্টা"=60, "প্রায় ৪ ঘণ্টা"=240, "৩ ঘন্টার অধিক"=180).
   stated_delay_reference = "from_origin" if the delay is measured from the departure station
   ("ঢাকা থেকে এক ঘন্টা বিলম্ব"), else "en_route".
6. locomotive_number: the 4-digit number (English digits) of KALNI's own engine, ONLY when the post ties it to Kalni:
   written like "2936/774", "773/2936", "২৯০১ & ৭৭৪ কালনী", or next to ইঞ্জিন/লোকো in a post about Kalni.
   The digits must appear in the post exactly. NEVER take: a number that belongs to another train
   ("3022 ইঞ্জিনের পারাবত", "2709/781"), a year (2026), a clock time, a count of people, a coach number.
   If a post lists engines for both Kalni trains, return only the one for this post's train, null if unclear.
   Take time to reason if necessary. Do NOT judge whether an engine is strong or weak: opinions about engines are not extracted.
7. driver_name: the loco master / driver of Kalni as written (এল এম, চালক, "চালকের দায়িত্বে আছেন"). Name only,
   without honorifics (ভাই, নানাভাই, প্রিয়). Null if not named. A name after 📸 / ছবি / ভিডিও / ©
   is a PHOTO CREDIT, never the driver.
8. crossings: each crossing with another train: other_train as written (পারাবত, উদয়ন, জয়ন্তিকা, উপবন, সুবর্ণা, উপকূল,
   ব্রহ্মপুত্র, তুরাগ, চট্টলা, or a train number), station, wait_minutes if stated.
   kalni_waited=true if Kalni was the one held (কালনী বসা / বসে আছে / লুপে), false if the other train waited, null if unclear.
9. overtaken_by: each other train given priority over Kalni: dispatched ahead of it from a station, or passed it
   while Kalni waited ("আগে যাবে", "পেছন থেকে আগে যাবে", "ফাঁকি দিয়ে ... আগে", "আগে ছেড়ে গেল", "আগে পাস").
   other_train, station, wait_minutes if stated. A normal crossing goes in crossings, not here.
10. delay_reasons: zero or more of {REASONS}. Use only what the post states or clearly implies.
11. Absent -> null. Never invent.
"""


def prep_text(t):
    t = unicodedata.normalize("NFC", t or "").replace("#", "").replace("_", " ")
    return "\n".join(re.sub(r"[ \t]+", " ", ln).strip() for ln in t.splitlines()).strip()


def post_key(p):
    return hashlib.md5(f"{SCHEMA_VERSION}|{p.get('date_time', '')}|{p.get('text', '')}".encode("utf-8")).hexdigest()


def call_gemini(batch):
    parts = []
    for i, p in enumerate(batch):
        dt = parse_post_dt(p.get("date_time"))
        stamp = dt.strftime("%A %Y-%m-%d %H:%M") if dt else "unknown"
        parts.append(f"### POST {i} | posted {stamp}\n{prep_text(p.get('text', ''))}")
    prompt = "Extract data for every post below.\n\n" + "\n\n".join(parts)

    for model in MODELS:
        delay = 4
        for attempt in range(1, 4):  # 3 attempts per model is plenty
            try:
                # print heartbeat so you know it's actively waiting for Google
                print(f"      -> querying {model} (attempt {attempt})...", flush=True)

                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=list[PostExtraction],
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.0,
                        # 45-second timeout prevents indefinite socket hangs
                        http_options=types.HttpOptions(timeout=45000),
                    ),
                )
                if resp.parsed:
                    return resp.parsed
                print(f"   [!] {model}: empty response", flush=True)
                break
            except Exception as e:
                err = str(e)
                print(f"   [!] {model} attempt {attempt}: {err[:120]}...", flush=True)

                # If daily quota is hit, do not sleep - move to next model immediately
                if "per_day" in err or "Daily" in err:
                    print(f"   [!] {model} daily quota exhausted. Skipping model.", flush=True)
                    break

                if any(c in err for c in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DeadlineExceeded", "timed out")):
                    time.sleep(delay)
                    delay = min(delay * 2, 25)
                else:
                    break
    return None


def extract_all(posts):
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    todo = [(post_key(p), p) for p in posts if post_key(p) not in cache]
    print(f"[+] {len(posts) - len(todo)} posts cached, {len(todo)} to extract", flush=True)
    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i + BATCH_SIZE]
        print(f"    extracting {i}-{i + len(chunk)} / {len(todo)}", flush=True)
        parsed = call_gemini([p for _, p in chunk])
        if parsed is None:
            continue                      # not cached -> retried on the next run
        for ex in parsed:
            if 0 <= ex.post_index < len(chunk):
                cache[chunk[ex.post_index][0]] = ex.model_dump()
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(4)
    return [cache.get(post_key(p)) for p in posts]


# =========================================================================
# OBSERVATIONS -> TRAIN / JOURNEY ASSIGNMENT
# =========================================================================
VALID_EVENTS = {"arrival", "departure", "pass", "halt", "approaching", "waiting"}
ARR_EVENTS = {"arrival", "pass", "halt", "approaching"}
ANY_EVENTS = ARR_EVENTS | {"departure"}


def sched_minutes(train, station, event, strict):
    e = TT[train].get(station)
    if not e:
        return None
    if event in ("departure", "waiting"):
        v = e["dep_m"]
    elif event == "arrival":
        v = e["arr_m"]
    else:
        v = e["arr_m"] if e["arr_m"] is not None else e["dep_m"]
    if v is None and not strict:
        v = e["arr_m"] if e["arr_m"] is not None else e["dep_m"]
    return v


def norm_train(x):
    s = str(x or "").translate(BN_TO_EN)
    return next((t for t in ("773", "774") if t in s), None)


def assign_train(station, event, actual, hint_train, hint_dir):
    explicit = hint_train or DIR_TO_TRAIN.get(hint_dir)
    cands = []
    for t in ((explicit,) if explicit else ("773", "774")):
        sm = sched_minutes(t, station, event, strict=not explicit)
        if sm is None:
            continue
        jd = journey_date_for(t, actual)
        delay = (actual - (day0(jd) + timedelta(minutes=sm))).total_seconds() / 60
        cands.append((t, jd, delay))
    if not explicit:
        cands = [c for c in cands if MIN_DELAY_MIN <= c[2] <= MAX_DELAY_MIN]
    return min(cands, key=lambda c: abs(c[2])) if cands else None


def guess_journey(post_dt, hint_train, hint_dir):
    t = hint_train or DIR_TO_TRAIN.get(hint_dir)
    if not t:
        mins = post_dt.hour * 60 + post_dt.minute
        t = "774" if 360 <= mins < 810 else "773"
    return t, journey_date_for(t, post_dt)


def _dedup(events):
    out = {}
    for e in events:
        k = (e["train"], e["station"])
        if k not in out:
            out[k] = dict(e)
        else:
            cur = out[k]
            if e.get("wait") and (not cur.get("wait") or e["wait"] > cur["wait"]):
                cur["wait"] = e["wait"]
            if cur.get("kalni_waited") is None:
                cur["kalni_waited"] = e.get("kalni_waited")
    return list(out.values())


def _norm_event(c):
    st_raw = (c.get("station") or "").strip()
    return {"train": canon_train(c.get("other_train")), "station": canonical_station(st_raw) or st_raw or None,
            "wait": c.get("wait_minutes"), "kalni_waited": c.get("kalni_waited")}

LOCO_NUM = r"(2[6-9]\d{2}|3[0-3]\d{2})"      # BR engine numbers; a year (20xx) or a time can never match
KALNI_NUM = r"(77[34])"
# separators that really bind an engine to a train: "2936/774", "2901 & 774", "2901 ও 774", "774 - 2936"
# ('*' is deliberately NOT one: in "2936/774*2709/781" the 2709 belongs to 781)
_SEP = r"\s*(?:[/&+,\-–—]|এবং|ও|আর|নং|নম্বর)?\s*"

_OTHER_TRAIN_TOKENS = [_nm(a.translate(BN_TO_EN)) for al in TRAIN_ALIASES.values() for a in al]


def extract_kalni_locos_from_text(raw_text: str) -> dict:
    """{'774': '2936', '773': '2908'}: only engines written right next to a Kalni train number (or ইঞ্জিন/লোকো
    on the same line). Never guesses; anything else is left to the (checked) Gemini fallback."""
    if not raw_text:
        return {}
    t = unicodedata.normalize("NFC", raw_text).translate(BN_TO_EN)
    out = {}
    # engine then train:  2936/774 | 2901 & 774
    for m in re.finditer(rf"(?<!\d){LOCO_NUM}(?!\d){_SEP}{KALNI_NUM}(?!\d)", t):
        out.setdefault(m.group(2), m.group(1))
    # train then engine:  774/2936 | 773 - 2908   (not when that engine is itself tied to another train: "774 & 2709/781")
    for m in re.finditer(rf"(?<!\d){KALNI_NUM}(?!\d){_SEP}{LOCO_NUM}(?!\d)(?!\s*/\s*\d{{2,3}})", t):
        out.setdefault(m.group(1), m.group(2))
    # keyword on the same line: "774 কালনী ইঞ্জিন ২৯০১" / "773 loco: 2921"
    for tr in ("773", "774"):
        if tr not in out:
            m = re.search(rf"(?<!\d){tr}(?!\d)[^\n\d]{{0,25}}?(?:ইঞ্জিন|লোকো|loco|engine)[^\d\n]{{0,8}}{LOCO_NUM}(?!\d)", t, re.I)
            if m:
                out[tr] = m.group(1)
    return out


def _tied_to_other_train(t: str, num: str) -> bool:
    """True if EVERY occurrence of `num` in the text sits next to another train (name or number)."""
    occ = [m for m in re.finditer(rf"(?<!\d){num}(?!\d)", t)]
    if not occ:
        return True
    for m in occ:
        before, after = t[max(0, m.start() - 22):m.start()], t[m.end():m.end() + 22]
        slash_other = re.match(r"\s*/\s*(\d{2,3})(?!\d)", after) or re.search(r"(?<!\d)(\d{2,3})\s*/\s*$", before)
        if slash_other and slash_other.group(1) not in ("773", "774"):
            continue
        near = _nm(before + " " + after)
        if any(tok and tok in near for tok in _OTHER_TRAIN_TOKENS):
            continue
        return False                       # this occurrence is not tied to another train
    return True


def gemini_loco(ex, raw_text):
    """Gemini's engine number is accepted only if it is in range, literally in the post, and not another train's."""
    t = unicodedata.normalize("NFC", raw_text or "").translate(BN_TO_EN)
    nums = {n for n in re.findall(rf"(?<!\d){LOCO_NUM}(?!\d)", str(ex.get("locomotive_number") or "").translate(BN_TO_EN))}
    ok = [n for n in nums if not _tied_to_other_train(t, n)]
    return ok[0] if len(ok) == 1 else None


CREDIT_LINE_RE = re.compile(r"(?mi)^[^\n]*(?:📸|©|ছবি\s*[-:–]|ভিডিও\s*[-:–]|photo\s*[-:–]|video\s*[-:–]|credit)[^\n]*$")


def is_photo_credit(driver_raw, text) -> bool:
    ct = _nm(" ".join(m.group(0) for m in CREDIT_LINE_RE.finditer(text or "")))
    toks = [w for w in (canon_driver(driver_raw) or "").split() if len(w) > 2]
    return bool(ct and toks and all(_nm(w) in ct for w in toks))


def make_meta(ex, post_dt, raw_text: str = "", target_train: Optional[str] = None, url: str = ""):
    explicit = extract_kalni_locos_from_text(raw_text)
    loco, method = None, None
    if target_train in explicit:
        loco, method = explicit[target_train], "text_explicit"
    elif not explicit:                                  # no engine tied to a Kalni train: use Gemini, but checked
        loco = gemini_loco(ex, raw_text)
        method = "text_inferred" if loco else None

    drv_raw = ex.get("driver_name")
    driver = None if (drv_raw and is_photo_credit(drv_raw, raw_text)) else canon_driver(drv_raw)
    crossings = [e for e in (_norm_event(c) for c in ex.get("crossings") or []) if e["train"] or e["station"]]
    overtakes = [e for e in (_norm_event(c) for c in ex.get("overtaken_by") or []) if e["train"] or e["station"]]
    reasons = [r for r in ex.get("delay_reasons") or [] if r in REASONS]
    stated = ex.get("stated_delay_minutes")

    if not any([loco, driver, crossings, overtakes, reasons, stated is not None]):
        return None

    return {
        "post_dt": post_dt, "locos": [loco] if loco else [], "loco_method": method, "driver": driver,
        "crossings": crossings, "overtakes": overtakes, "reasons": reasons,
        "stated": stated, "stated_ref": ex.get("stated_delay_reference"),
        "url": url, "snippet": " ".join((raw_text or "").split())[:90],
    }


def build_events(posts, extractions):
    obs_by_j = defaultdict(list)
    meta_by_j = defaultdict(list)
    pending = []
    no_date = 0

    for p, ex in zip(posts, extractions):
        if not ex or not ex.get("is_relevant"):
            continue
        post_dt = parse_post_dt(p.get("date_time"))
        if not post_dt:
            no_date += 1
            continue

        raw_text = p.get("text", "")
        hint_t = norm_train(ex.get("train_number"))
        hint_d = (ex.get("direction") or "").strip().lower()

        # --- 1. Process Observations (Arrival/Departure/Pass) ---
        keys = []
        for o in ex.get("observations") or []:
            event = (o.get("event") or "").strip().lower()
            st = canonical_station(o.get("station"))
            clock = parse_clock(o.get("time_text"))
            if event not in VALID_EVENTS or not st or not clock:
                continue

            actual = resolve_actual_dt(post_dt, *clock)
            res = assign_train(st, event, actual, hint_t, hint_d)
            if not res:
                continue

            train, jd, delay = res
            obs_by_j[(train, jd)].append({
                "station": st, "event": event, "actual": actual, "delay": delay,
                "post_dt": post_dt, "is_forecast": bool(o.get("is_forecast")), "url": p.get("url", ""),
            })
            keys.append((train, jd))

        # --- 2. Which Kalni train(s) does the post talk about? ---
        txt_norm = raw_text.translate(BN_TO_EN)
        nm = _nm(raw_text)                      # spaces removed: "ঢাকা গামী" == "ঢাকাগামী"
        trains_in_post = set()
        if re.search(r"(?<!\d)773(?!\d)", txt_norm) or any(x in nm for x in ("ঢাকাটুসিলেট", "সিলেটগামী", "সিলেটঅভিমুখী", "লন্ডনগামী", "সিলেটেরউদ্দেশ্যে")):
            trains_in_post.add("773")
        if re.search(r"(?<!\d)774(?!\d)", txt_norm) or any(x in nm for x in ("সিলেটটুঢাকা", "ঢাকাগামী", "ঢাকাঅভিমুখী", "ঢাকামুখী", "ঢাকারউদ্দেশ্যে")):
            trains_in_post.add("774")

        if not trains_in_post and keys:         # no explicit train: use where the timed events landed
            trains_in_post.add(Counter([k[0] for k in keys]).most_common(1)[0][0])
        if not trains_in_post:                  # last resort: hint or time of day
            fallback_t = hint_t or DIR_TO_TRAIN.get(hint_d)
            if not fallback_t:
                mins = post_dt.hour * 60 + post_dt.minute
                fallback_t = "774" if 360 <= mins < 810 else "773"
            trains_in_post.add(fallback_t)

        # --- 3. Attach metadata strictly per train ---
        for tr in trains_in_post:
            meta = make_meta(ex, post_dt, raw_text=raw_text, target_train=tr, url=p.get("url", ""))
            if meta is None:
                continue
            matching_keys = [k for k in keys if k[0] == tr]
            if len(trains_in_post) > 1 and not matching_keys:
                # post names both trains and nothing places it: keep ONLY the engine tied to this train
                meta = dict(meta, driver=None, crossings=[], overtakes=[], reasons=[], stated=None)
                if not meta["locos"]:
                    continue
            if matching_keys:
                meta_by_j[Counter(matching_keys).most_common(1)[0][0]].append(meta)
            else:
                pending.append((meta, tr, hint_d))

    # --- 4. Resolve Untimed Posts within Matching Journey Window ---
    spans = {k: (min(o["actual"] for o in v) - timedelta(minutes=90), max(o["actual"] for o in v) + timedelta(minutes=90))
             for k, v in obs_by_j.items()}

    for meta, tr, hint_d in pending:
        cands = [k for k, (a, b) in spans.items() if a <= meta["post_dt"] <= b and k[0] == tr]
        if cands:
            key = min(cands, key=lambda k: abs((spans[k][0] + (spans[k][1] - spans[k][0]) / 2) - meta["post_dt"]))
        else:
            key = (tr, journey_date_for(tr, meta["post_dt"]))
        meta_by_j[key].append(meta)

    if no_date:
        print(f"[!] {no_date} relevant posts skipped: no post timestamp (re-scrape them)", flush=True)
    return obs_by_j, meta_by_j


# =========================================================================
# PER-JOURNEY REASONING  (stack = newest post first)
# =========================================================================
def prepare_obs(obs):
    obs = sorted(obs, key=lambda o: o["post_dt"], reverse=True)
    good = [o for o in obs if MIN_DELAY_MIN <= o["delay"] <= MAX_DELAY_MIN]
    if len(good) >= 4:
        med = median(o["delay"] for o in good)
        good = [o for o in good if abs(o["delay"] - med) <= OUTLIER_MIN]
    return good


def pick(obs, station, events):
    """Newest actual observation first; forecasts only if no actual exists."""
    for want_fc in (False, True):
        for o in obs:
            if o["station"] == station and o["event"] in events and o["is_forecast"] == want_fc:
                return o
    return None


def ref_m(train, st):
    e = TT[train][st]
    return e["arr_m"] if e["arr_m"] is not None else e["dep_m"]


def direct_delays(train, obs):
    """Delay (min) at every station with a real (non-forecast) post."""
    order, delays = ROUTE[train], {}
    for st in order:
        o = pick(obs, st, {"departure"} if st == order[0] else ANY_EVENTS)
        if o and not o["is_forecast"]:
            delays[st] = max(0.0, o["delay"]) if st == order[0] else o["delay"]
    return delays


def compute_gain_rates(journeys):
    """Delay stacking: median minutes of delay gained per scheduled minute of running, per train,
    from every pair of directly-reported stations on the same journey."""
    pairs = defaultdict(list)
    for (train, jd), (obs, _) in journeys.items():
        pts = sorted((ref_m(train, st), d) for st, d in direct_delays(train, obs).items())
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                dt = pts[j][0] - pts[i][0]
                if dt >= 30:
                    pairs[train].append((pts[j][1] - pts[i][1]) / dt)
    rates = {}
    for t in ("773", "774"):
        v = pairs.get(t, [])
        rates[t] = float(min(0.5, max(-0.05, median(v)))) if len(v) >= 5 else 0.0
    return rates


def project_delay(train, src, dst, d_src, rate):
    """Carry the delay seen at `src` to `dst`, allowing it to grow (or shrink backwards) at `rate`."""
    return max(0.0, d_src + rate * (ref_m(train, dst) - ref_m(train, src)))


def nearest(train, target, delays):
    order = ROUTE[train]
    ti, tref = order.index(target), ref_m(train, target)
    best = None
    for st, d in delays.items():
        if st == target:
            continue
        key = (abs(ref_m(train, st) - tref), 0 if order.index(st) < ti else 1)  # tie -> previous station
        if best is None or key < best[0]:
            best = (key, st, d)
    return best  # ((dist_min, side), station, delay)


def resolve_departure(train, jd, obs, metas, rate):
    order = ROUTE[train]
    origin = order[0]
    sched_dep = TT[train][origin]["dep_m"]
    base = day0(jd)

    dep_o = pick(obs, origin, {"departure"})
    delay, src = None, None
    if dep_o and not dep_o["is_forecast"]:
        delay, src = max(0.0, dep_o["delay"]), "direct"
    elif dep_o:
        delay, src = max(0.0, dep_o["delay"]), "forecast_post"
    else:
        stated = next((m for m in metas if m["stated"] is not None and m["stated_ref"] == "from_origin"), None)
        if stated:
            delay, src = float(stated["stated"]), "stated_in_post"
        else:
            down = [o for st in order[1:] for o in [pick(obs, st, ANY_EVENTS)] if o and not o["is_forecast"]]
            if down:
                d1, st1 = down[0]["delay"], down[0]["station"]   # first reported station after the origin
                if d1 <= ON_TIME_TOL_MIN:
                    delay, src = 0.0, "inferred_on_time"
                else:
                    delay, src = project_delay(train, st1, origin, d1, rate), f"estimated_from_{st1}"
    wait_o = pick(obs, origin, {"waiting"})      # "still waiting for engine" = lower bound
    if wait_o and (delay is None or wait_o["delay"] > delay) and src != "direct":
        delay, src = max(0.0, wait_o["delay"]), "waiting_lower_bound"

    if delay is None:
        return {"delay": None, "source": None, "dt": None}
    return {"delay": delay, "source": src, "dt": base + timedelta(minutes=sched_dep + delay)}


def pick_loco(ms):
    """Engine with the most evidence (explicit '2936/774' counts 3, an inferred mention 1); ties -> earliest mention."""
    w, first = Counter(), {}
    for m in ms:
        for l in m["locos"]:
            w[l] += 3 if m.get("loco_method") == "text_explicit" else 1
            first[l] = min(first.get(l, m["post_dt"]), m["post_dt"])
    if not w:
        return None
    return max(w, key=lambda l: (w[l], -first[l].timestamp()))


def agg_meta(metas, dep_dt, origin):
    uniq = lambda xs: list(dict.fromkeys(x for x in xs if x))
    cutoff = dep_dt + timedelta(minutes=10) if dep_dt else None
    pre = [m for m in metas if cutoff and m["post_dt"] <= cutoff]     # posts made before the train left

    def top(ms, getter):
        vals = [v for m in ms for v in getter(m)]
        return Counter(vals).most_common(1)[0][0] if vals else None

    g_drv = lambda m: [m["driver"]] if m["driver"] else []
    loco_pre, loco_any = pick_loco(pre), pick_loco(metas)
    drv_pre, drv_any = top(pre, g_drv), top(metas, g_drv)
    loco_primary, driver_primary = loco_pre or loco_any, drv_pre or drv_any
    explicit = uniq(l for m in metas if m.get("loco_method") == "text_explicit" for l in m["locos"])
    loco_method = None
    if loco_primary:
        loco_method = "text_explicit" if loco_primary in explicit else "text_inferred"

    crossings = _dedup([c for m in metas for c in m["crossings"]])
    overtakes = _dedup([c for m in metas for c in m["overtakes"]])
    waited = [c for c in crossings if c["kalni_waited"] is True]
    reasons = {r for m in metas for r in m["reasons"]}
    if waited:
        reasons.add("crossing_wait")
    if overtakes:
        reasons.add("overtaken_by_other_train")
    wait_total = sum(c["wait"] or 0 for c in crossings if c["kalni_waited"] is not False)
    ot_wait = sum(c["wait"] or 0 for c in overtakes)

    return {
        # engine (strength / series are derived from the number later, never from opinions in posts)
        "loco_primary": loco_primary, "loco_method": loco_method,
        "locomotive_numbers": "; ".join(explicit or ([loco_primary] if loco_primary else [])) or None,
        "loco_changed": int(len(explicit) > 1),      # only when 2+ numbers are each explicitly tied to this train
        "loco_pre_departure": loco_pre,
        # crew
        "driver_primary": driver_primary, "driver_names": "; ".join(uniq(m["driver"] for m in metas)) or None,
        "driver_pre_departure": drv_pre,
        # crossings
        "crossing_count": len(crossings),
        "crossing_trains": "; ".join(uniq(c["train"] for c in crossings)) or None,
        "crossing_stations": "; ".join(uniq(c["station"] for c in crossings)) or None,
        "kalni_waited_crossings": len(waited),
        "crossing_wait_total_min": wait_total or None,
        "max_crossing_wait_min": max([c["wait"] for c in crossings if c["wait"]], default=None),
        # priority given to other trains
        "overtaken_count": len(overtakes),
        "overtaken_by": "; ".join(uniq(c["train"] for c in overtakes)) or None,
        "overtaken_wait_total_min": ot_wait or None,
        "overtaken_at_origin": int(any(c["station"] == origin for c in overtakes)),
        "trains_given_priority": "; ".join(uniq([c["train"] for c in waited] + [c["train"] for c in overtakes])) or None,
        # causes
        "vip_movement": int("vip_or_special_movement" in reasons),
        "delay_reasons": "; ".join(sorted(reasons)) or None,
        **{f"reason_{r}": int(r in reasons) for r in REASONS},
    }


def label_weight(src, dist):
    if src == "direct":
        return 1.0
    if src == "forecast_post":
        return 0.8
    if src == "estimated_from_next_departure":
        return 0.5
    if src and src.startswith("estimated_from_"):
        return round(max(0.2, 1 - (dist or 0) / 400), 2)
    return None


def resolve_journey(train, jd, obs, metas, dep_info, dep_ctx, rate):
    tt, order = TT[train], ROUTE[train]
    origin, dest = order[0], order[-1]
    base = day0(jd)
    delays = direct_delays(train, obs)

    def estimate_arrival_delay(station, is_dest):
        """-> (delay, source, estimated_from_station, distance_min)"""
        o = pick(obs, station, ANY_EVENTS)
        if o and not o["is_forecast"]:
            return o["delay"], "direct", None, 0
        if o:
            return o["delay"], "forecast_post", None, 0
        near = nearest(train, station, delays)
        if near and near[0][0] <= NEAR_STATION_MIN:
            return (project_delay(train, near[1], station, near[2], rate),
                    f"estimated_from_{near[1]}", near[1], near[0][0])
        if is_dest and train == "774":            # rake turnaround: next 773 departure from Dhaka
            d773 = dep_ctx.get(jd)
            if d773 and d773["delay"] is not None and d773["delay"] >= 45 and d773["source"] != "inferred_on_time":
                arr_dt = d773["dt"] - timedelta(minutes=TURNAROUND_MIN)
                sched_dt = base + timedelta(minutes=tt[station]["arr_m"])
                return (max(0.0, (arr_dt - sched_dt).total_seconds() / 60),
                        "estimated_from_next_departure", None, None)
        if near:
            return (project_delay(train, near[1], station, near[2], rate),
                    f"estimated_from_{near[1]}", near[1], near[0][0])
        return None, None, None, None

    sh_delay, sh_src, sh_from, sh_dist = estimate_arrival_delay("Shaistaganj", False)
    de_delay, de_src, de_from, de_dist = estimate_arrival_delay(dest, True)

    def arr_dt(station, delay):
        return base + timedelta(minutes=tt[station]["arr_m"] + delay) if delay is not None else None

    sh_dt, de_dt = arr_dt("Shaistaganj", sh_delay), arr_dt(dest, de_delay)
    rnd = lambda x: None if x is None else int(round(x))

    # delay stacking from DIRECT posts only: where along the route did time get lost / recovered?
    seq = [(st, delays[st]) for st in order if st in delays]
    deltas = [(a[0], b[0], b[1] - a[1]) for a, b in zip(seq, seq[1:])]
    worst = max(deltas, key=lambda x: x[2]) if deltas else None
    dep_d = dep_info["delay"]

    row = {
        "journey_date": jd.isoformat(), "journey_day": jd.strftime("%A"), "train_number": train,
        "direction": "Dhaka to Sylhet" if train == "773" else "Sylhet to Dhaka",
        "scheduled_departure": mm_to_hhmm(tt[origin]["dep_m"]),
        "scheduled_shaistaganj_arrival": mm_to_hhmm(tt["Shaistaganj"]["arr_m"]),
        "scheduled_destination_arrival": mm_to_hhmm(tt[dest]["arr_m"]),
        "departure_time": t24(dep_info["dt"]), "departure_delay_min": rnd(dep_d),
        "departure_source": dep_info["source"],
        "shaistaganj_arrival_time": t24(sh_dt), "shaistaganj_delay_min": rnd(sh_delay),
        "shaistaganj_source": sh_src, "shaistaganj_estimate_from": sh_from,
        "shaistaganj_estimate_distance_min": sh_dist,
        "destination_arrival_time": t24(de_dt),
        "destination_arrival_date": de_dt.date().isoformat() if de_dt else None,
        "destination_delay_min": rnd(de_delay), "destination_source": de_src,
        "destination_estimate_from": de_from, "destination_estimate_distance_min": de_dist,
        "destination_label_weight": label_weight(de_src, de_dist),
        "is_late": None if de_delay is None else int(de_delay > ON_TIME_TOL_MIN),
        # delay stacking
        "gain_origin_to_shaistaganj": rnd(sh_delay - dep_d) if sh_delay is not None and dep_d is not None else None,
        "gain_shaistaganj_to_destination": rnd(de_delay - sh_delay) if de_delay is not None and sh_delay is not None else None,
        "direct_stations_reported": len(seq),
        "direct_max_gain_min": rnd(worst[2]) if worst and worst[2] > 0 else 0,
        "direct_max_gain_segment": f"{worst[0]}->{worst[1]}" if worst and worst[2] > 0 else None,
        "direct_recovered_min": rnd(-sum(d[2] for d in deltas if d[2] < 0)) if deltas else None,
        "n_observations": len(obs),
        "n_posts": len({o["url"] or o["post_dt"] for o in obs}),
    }
    row.update(agg_meta(metas, dep_info["dt"], origin))
    return row


def panel_rows(train, jd, obs):
    """Long table: one row per directly-reported station (for a station-level nowcast model)."""
    dl, order, rows, prev = direct_delays(train, obs), ROUTE[train], [], None
    for i, st in enumerate(order):
        if st in dl:
            rows.append({"journey_date": jd.isoformat(), "train_number": train, "station": st, "station_idx": i,
                         "sched_min": ref_m(train, st), "delay_min": round(dl[st]),
                         "prev_reported_station": prev[0] if prev else None,
                         "delta_from_prev_reported_min": None if prev is None else round(dl[st] - prev[1])})
            prev = (st, dl[st])
    return rows


# =========================================================================
# POST-PROCESSING: history features, cascade, dynamic flags, column guide
# =========================================================================
def add_history_encoding(df, key_col, prefix):
    """Mean destination delay of earlier journeys (strictly earlier dates) with the same driver / engine."""
    hist = defaultdict(list)
    means, ns = {}, {}
    for date in sorted(df["journey_date"].unique()):
        idx = df.index[df["journey_date"] == date]
        for i in idx:
            k = df.at[i, key_col]
            h = hist.get(k) if isinstance(k, str) and k else None
            means[i], ns[i] = (round(sum(h) / len(h), 1), len(h)) if h else (None, 0)
        for i in idx:
            k, y = df.at[i, key_col], df.at[i, "destination_delay_min"]
            if isinstance(k, str) and k and not pd.isna(y):
                hist[k].append(float(y))
    df[f"{prefix}_prev_mean_delay_min"] = pd.Series(means)
    df[f"{prefix}_prev_n"] = pd.Series(ns)


def add_flag_columns(df, col, prefix):
    counts = Counter(t for v in df[col].dropna() for t in str(v).split("; ") if t)
    for t, n in counts.items():
        if n >= MIN_FLAG_COUNT:
            df[prefix + re.sub(r"\W+", "_", t)] = df[col].fillna("").apply(lambda v, t=t: int(t in v.split("; ")))


def merge_image_locos(df):
    """Fill engine columns from output/engine_from_images.csv (made by engine_from_images.py).
    Only journeys with NO engine number in the post text are filled; text always wins."""
    df["loco_source"] = df["loco_primary"].apply(lambda v: "text" if isinstance(v, str) and v else None)
    df["loco_image_votes"], df["loco_image_score"] = None, None
    f = OUT / "engine_from_images.csv"
    if not f.exists():
        return df
    img = pd.read_csv(f, dtype=str).fillna("")
    img = img[img["loco_from_image"] != ""]
    found = {(r.journey_date, r.train_number): r for r in img.itertuples()}
    n_filled = 0
    for i in df.index:
        r = found.get((df.at[i, "journey_date"], df.at[i, "train_number"]))
        if r is None or df.at[i, "loco_source"] == "text":
            continue
        n = r.loco_from_image
        if not (n.isdigit() and LOCO_MIN <= int(n) <= LOCO_MAX):      # a year / sign read by mistake
            continue
        df.at[i, "loco_primary"], df.at[i, "locomotive_numbers"] = n, n
        df.at[i, "loco_series"] = f"{n[:2]}xx" if len(n) == 4 else None
        df.at[i, "loco_source"] = "image"
        if getattr(r, "loco_image_pre_departure", "") == "1":      # photo taken before the train left
            df.at[i, "loco_pre_departure"] = n
        df.at[i, "loco_image_votes"], df.at[i, "loco_image_score"] = int(r.loco_image_votes or 0), float(r.loco_image_score or 0)
        n_filled += 1
    print(f"[+] engine numbers filled from images: {n_filled}", flush=True)
    return df


def add_engine_features(df):
    """Everything about the engine comes from its NUMBER. No number -> all of these stay empty (never 0 / 'standard')."""
    nums = [int(v) if isinstance(v, str) and v.isdigit() else None for v in df["loco_primary"]]
    df["loco_series"] = [f"{str(n)[:2]}xx" if n else None for n in nums]
    df["loco_strength"] = [None if n is None else ("strong" if n >= STRONG_LOCO_MIN else "standard") for n in nums]
    df["is_strong_engine"] = pd.array([None if n is None else int(n >= STRONG_LOCO_MIN) for n in nums], dtype="Int64")
    df["loco_type"] = [LOCO_TYPE_MAP.get(v) if isinstance(v, str) else None for v in df["loco_primary"]]
    return df


def stage_of(col):
    if col in ("n_observations", "n_posts"):
        return "bookkeeping"
    if col.startswith(("destination_", "is_late")):
        return "outcome"
    if col in ("journey_date", "journey_day", "train_number", "direction") or col.startswith(("scheduled_", "prev_same", "rolling3", "inbound_")):
        return "pre_day"
    if col.endswith("_pre_departure"):
        return "pre_departure"
    if col.startswith(("driver_prev", "loco_prev")):
        return "pre_departure_if_driver_or_engine_known"
    return "in_journey"       # departure_*, shaistaganj_*, gain_*, direct_*, crossing*, overtaken*, reason_*, driver_*, loco_*, engine_*


def process_full_dataset(json_file_path="output/posts.json"):
    raw = json.loads(Path(json_file_path).read_text(encoding="utf-8"))
    seen, posts = set(), []
    for p in raw:                                # de-duplicate
        k = post_key(p)
        if k not in seen and (p.get("text") or "").strip():
            seen.add(k)
            posts.append(p)
    print(f"[+] {len(posts)} unique posts", flush=True)

    extractions = extract_all(posts)
    obs_by_j, meta_by_j = build_events(posts, extractions)

    audit = [{"train_number": t, "journey_date": jd.isoformat(), **{k: v for k, v in o.items() if k != "post_dt"},
              "actual": o["actual"].strftime("%Y-%m-%d %H:%M"), "post_time": o["post_dt"].strftime("%Y-%m-%d %H:%M"),
              "delay": round(o["delay"])} for (t, jd), lst in obs_by_j.items() for o in lst]
    pd.DataFrame(audit).to_csv(OUT / "all_observations.csv", index=False, encoding="utf-8-sig")

    journeys = {}
    for (train, jd), obs in obs_by_j.items():
        if jd.weekday() == 4:                    # Friday = weekly off day
            continue
        metas = sorted(meta_by_j.get((train, jd), []), key=lambda m: m["post_dt"], reverse=True)
        journeys[(train, jd)] = (prepare_obs(obs), metas)

    rates = compute_gain_rates(journeys)
    print(f"[+] delay stacking (min gained per scheduled minute): 773={rates['773']:.3f}  774={rates['774']:.3f}", flush=True)

    dep_info = {k: resolve_departure(k[0], k[1], o, m, rates[k[0]]) for k, (o, m) in journeys.items()}
    dep_ctx = {jd: info for (t, jd), info in dep_info.items() if t == "773"}

    rows, panel = [], []
    for (t, jd), (o, m) in journeys.items():
        if not o:
            continue
        rows.append(resolve_journey(t, jd, o, m, dep_info[(t, jd)], dep_ctx, rates[t]))
        panel += panel_rows(t, jd, o)
    df = pd.DataFrame(rows).sort_values(["journey_date", "train_number"]).reset_index(drop=True)
    df = merge_image_locos(df)
    df = add_engine_features(df)

    # ---- lags and rake cascade (what was known before the journey) ----
    df = df.sort_values(["train_number", "journey_date"])
    g = df.groupby("train_number")["destination_delay_min"]
    df["prev_same_train_delay_min"] = g.shift(1)
    df["rolling3_same_train_delay_min"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    lookup = {(r.journey_date, r.train_number): r.destination_delay_min for r in df.itertuples()}

    def inbound(r):
        d = datetime.strptime(r.journey_date, "%Y-%m-%d").date()
        if r.train_number == "773":      # the rake arrives as today's 774
            return lookup.get((d.isoformat(), "774")), SLACK_773_FROM_774
        for back in (1, 2):              # 774 uses yesterday's 773 rake (Friday off -> look back 2 days)
            v = lookup.get(((d - timedelta(days=back)).isoformat(), "773"))
            if v is not None or (d - timedelta(days=back)).weekday() != 4:
                return v, SLACK_774_FROM_773 + 1440 * (back - 1)
        return None, SLACK_774_FROM_773
    ib = [inbound(r) for r in df.itertuples()]
    df["inbound_rake_delay_min"] = [None if pd.isna(v) else v for v, _ in ib]
    df["inbound_slack_min"] = [s for _, s in ib]
    df["inbound_delay_over_slack_min"] = [None if pd.isna(v) else max(0, v - s) for v, s in ib]
    df = df.sort_values(["journey_date", "train_number"]).reset_index(drop=True)

    # ---- driver / engine track record, other trains that cross or overtake ----
    add_history_encoding(df, "driver_primary", "driver")
    add_history_encoding(df, "loco_primary", "loco")
    add_flag_columns(df, "crossing_trains", "crossed_")
    add_flag_columns(df, "overtaken_by", "overtaken_by_")

    df.to_csv(OUT / "kalni_journeys_all.csv", index=False, encoding="utf-8-sig")
    ml = df.dropna(subset=["is_late"])
    ml.to_csv(OUT / "kalni_train_delay_ml_dataset.csv", index=False, encoding="utf-8-sig")
    chosen = {(r.journey_date, r.train_number): r.loco_primary for r in df.itertuples()}
    ev = [{"journey_date": jd.isoformat(), "train_number": t, "number": l, "method": m.get("loco_method"),
           "chosen": int(chosen.get((jd.isoformat(), t)) == l), "post_time": m["post_dt"].strftime("%Y-%m-%d %H:%M"),
           "post_url": m.get("url", ""), "post_text": m.get("snippet", "")}
          for (t, jd), ms in meta_by_j.items() for m in ms for l in m["locos"]]
    pd.DataFrame(ev).to_csv(OUT / "locomotive_evidence.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(panel).to_csv(OUT / "station_delays.csv", index=False, encoding="utf-8-sig")
    (OUT / "column_guide.json").write_text(json.dumps({c: stage_of(c) for c in df.columns}, indent=1), encoding="utf-8")

    print(f"\n[SUCCESS] journeys: {len(df)} | with target: {len(ml)} | station rows: {len(panel)}")
    print(df["destination_source"].fillna("none").str.replace(r"estimated_from_(?!next).*", "estimated_from_station",
                                                              regex=True).value_counts().to_string())
    print("files: kalni_train_delay_ml_dataset.csv, kalni_journeys_all.csv, station_delays.csv, "
          "all_observations.csv, column_guide.json (in output/)")


if __name__ == "__main__":
    process_full_dataset("output/posts.json")