import hashlib
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Missing GEMINI_API_KEY! Set it in your environment or .env file.")
client = genai.Client(api_key=GEMINI_API_KEY)

MODELS = [m.strip() for m in os.getenv("GEMINI_MODELS", "gemini-3.5-flash-lite,gemini-3.8-flash,gemini-3.1-flash-lite").split(",")]
BATCH_SIZE = 15

OUT = Path("output")
OUT.mkdir(exist_ok=True)
CACHE_FILE = OUT / "extraction_cache.json"

BN_TO_EN = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# =========================================================================
# 1. OFFICIAL TIMETABLE & DISTANCE PROGRESSION (MINUTES FROM MIDNIGHT)
# =========================================================================
def _hm(s):
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def _build(rows):
    return [{"station": s, "arr_m": _hm(a) if a else None, "dep_m": _hm(d) if d else None, "dist_km": dist} for s, a, d, dist in rows]


T773 = _build([
    ("Dhaka", None, "14:55", 0), ("Biman Bandar", "15:17", "15:22", 21), ("Narsingdi", "15:55", "15:57", 57),
    ("Bhairab Bazar", "16:20", "16:22", 84), ("Brahmanbaria", "16:46", "16:49", 112), ("Azampur", "17:12", "17:14", 132),
    ("Harashpur", "17:34", "17:36", 149), ("Montola", "17:50", "17:52", 162), ("Shaistaganj", "18:12", "18:15", 183),
    ("Noapara", "18:25", "18:27", 195), ("Rashidpur", "18:35", "18:37", 205), ("Satgaon", "18:45", "18:47", 216),
    ("Sreemangal", "18:52", "18:55", 227), ("Bhanugach", "19:10", "19:12", 243), ("Shamshernagar", "19:22", "19:26", 254),
    ("Kulaura", "19:49", "19:51", 276), ("Baramchal", "20:05", "20:07", 290), ("Maijgaon", "20:20", "20:23", 305),
    ("Moglabazar", "20:55", "20:57", 319), ("Sylhet", "21:30", None, 335),
])

T774 = _build([
    ("Sylhet", None, "06:15", 0), ("Moglabazar", "06:35", "06:37", 16), ("Maijgaon", "06:55", "06:57", 30),
    ("Baramchal", "07:10", "07:12", 45), ("Kulaura", "07:23", "07:26", 59), ("Shamshernagar", "07:50", "07:52", 81),
    ("Bhanugach", "08:05", "08:07", 92), ("Sreemangal", "08:17", "08:20", 108), ("Satgaon", "08:30", "08:32", 119),
    ("Rashidpur", "08:42", "08:44", 130), ("Shaistaganj", "08:57", "09:00", 152), ("Noapara", "09:12", "09:14", 164),
    ("Montola", "09:30", "09:32", 173), ("Harashpur", "09:52", "09:54", 186), ("Azampur", "10:20", "10:22", 203),
    ("Brahmanbaria", "10:48", "10:51", 223), ("Bhairab Bazar", "11:10", "11:13", 251), ("Narsingdi", "11:43", "11:45", 278),
    ("Biman Bandar", "12:24", "12:28", 314), ("Dhaka", "12:55", None, 335),
])

TT = {"773": {e["station"]: e for e in T773}, "774": {e["station"]: e for e in T774}}
ROUTE = {"773": [e["station"] for e in T773], "774": [e["station"] for e in T774]}


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
    "Shaistaganj": ["shaistaganj", "shayestaganj", "শায়েস্তাগঞ্জ", "শায়েস্থাগঞ্জ"],
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
    return None


def classify_locomotive(loco_str: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not loco_str:
        return None, None
    m = re.search(r"\b(26\d{2}|29\d{2}|30\d{2}|31\d{2}|27\d{2})\b", str(loco_str))
    if not m:
        return str(loco_str).strip(), "Unknown"
    num = m.group(1)
    prefix = num[:2]
    if prefix == "26":
        return num, "Class_2600_Aging"
    elif prefix == "29":
        return num, "Class_2900_Standard"
    elif prefix in ("30", "31"):
        return num, "Class_3000_Monster"
    return num, "Other"


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
    hours = [h] if (explicit or h >= 13 or h == 0) else ([12, 0] if h == 12 else [h, h + 12])
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
    return (actual - timedelta(hours=6)).date() if train == "773" else actual.date()


def day0(d):
    return datetime.combine(d, datetime.min.time())


def t24(dt):
    return dt.strftime("%H:%M") if dt else None


def mm_to_hhmm(m):
    return f"{(m % 1440) // 60:02d}:{(m % 1440) % 60:02d}" if m is not None else None


# =========================================================================
# 2. GEMINI EXTRACTION WITH OPERATIONAL GROUNDING
# =========================================================================
class Obs(BaseModel):
    station: str
    event: str  # arrival | departure | pass | halt | approaching | waiting
    time_text: Optional[str] = None
    is_forecast: Optional[bool] = None


class Crossing(BaseModel):
    other_train: Optional[str] = None
    station: Optional[str] = None
    wait_minutes: Optional[int] = None


class PostExtraction(BaseModel):
    post_index: int
    is_operational: bool
    train_number: Optional[str] = None
    direction: Optional[str] = None
    observations: Optional[List[Obs]] = None
    stated_delay_minutes: Optional[int] = None
    stated_delay_reference: Optional[str] = None
    locomotive_number: Optional[str] = None
    driver_name: Optional[str] = None
    platform_number: Optional[str] = None
    crossings: Optional[List[Crossing]] = None
    priority_loss: Optional[bool] = None
    favored_train: Optional[str] = None
    engine_issue: Optional[bool] = None
    delay_reasons: Optional[List[str]] = None


SYSTEM_PROMPT = """
You extract railway operational metrics from informal Bengali Facebook admin posts about Kalni Express (Trains 773 & 774 on the Dhaka-Sylhet corridor).
Extract station events, clock times, crossing halts, engine numbers, and driver names accurately.
If not mentioned in the post, leave as null. Never hallucinate.
"""


def post_key(p):
    return hashlib.md5(f"{p.get('date_time', '')}|{p.get('text', '')}".encode("utf-8")).hexdigest()


def call_gemini(batch):
    parts = []
    for i, p in enumerate(batch):
        dt = parse_post_dt(p.get("date_time"))
        stamp = dt.strftime("%A %Y-%m-%d %H:%M") if dt else "unknown"
        txt = unicodedata.normalize("NFC", p.get("text") or "").replace("#", "").replace("_", " ")
        parts.append(f"### POST {i} | posted {stamp}\n{txt}")
    prompt = "Extract operational railway data for each post below.\n\n" + "\n\n".join(parts)

    for model in MODELS:
        delay = 4
        for attempt in range(1, 5):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json", response_schema=list[PostExtraction],
                        system_instruction=SYSTEM_PROMPT, temperature=0.0),
                )
                if resp.parsed:
                    return resp.parsed
                break
            except Exception as e:
                err = str(e)
                if any(c in err for c in ("503", "429", "UNAVAILABLE")):
                    time.sleep(delay)
                    delay = min(delay * 2, 40)
                else:
                    break
    return None


def extract_all(posts):
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.exists() else {}
    todo = [(post_key(p), p) for p in posts if post_key(p) not in cache]
    print(f"[+] {len(posts) - len(todo)} posts cached, {len(todo)} to query from Gemini", flush=True)

    for i in range(0, len(todo), BATCH_SIZE):
        chunk = todo[i:i + BATCH_SIZE]
        print(f"    processing batch {i // BATCH_SIZE + 1} ({i}-{i + len(chunk)})...", flush=True)
        parsed = call_gemini([p for _, p in chunk])
        if parsed is None:
            continue
        for ex in parsed:
            if 0 <= ex.post_index < len(chunk):
                cache[chunk[ex.post_index][0]] = ex.model_dump()
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
        time.sleep(4)
    return [cache.get(post_key(p)) for p in posts]


VALID_EVENTS = {"arrival", "departure", "pass", "halt", "approaching", "waiting"}


def sched_minutes(train, station, event):
    e = TT[train].get(station)
    if not e:
        return None
    if event in ("departure", "waiting"):
        return e["dep_m"] or e["arr_m"]
    return e["arr_m"] or e["dep_m"]


def assign_train(station, event, actual, hint_train, hint_dir):
    explicit = hint_train or ("773" if hint_dir == "dhaka_to_sylhet" else ("774" if hint_dir == "sylhet_to_dhaka" else None))
    cands = []
    for t in ((explicit,) if explicit else ("773", "774")):
        sm = sched_minutes(t, station, event)
        if sm is None:
            continue
        jd = journey_date_for(t, actual)
        delay = (actual - (day0(jd) + timedelta(minutes=sm))).total_seconds() / 60
        cands.append((t, jd, delay))
    if not explicit:
        cands = [c for c in cands if -45 <= c[2] <= 550]
    return min(cands, key=lambda c: abs(c[2])) if cands else None


def build_events(posts, extractions):
    obs_by_j = defaultdict(list)
    meta_by_j = defaultdict(list)

    for p, ex in zip(posts, extractions):
        if not ex or not ex.get("is_operational"):
            continue
        post_dt = parse_post_dt(p.get("date_time"))
        if not post_dt:
            continue

        hint_t = next((t for t in ("773", "774") if t in str(ex.get("train_number") or "").translate(BN_TO_EN)), None)
        hint_d = (ex.get("direction") or "").strip().lower()

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

        loco_num, loco_cls = classify_locomotive(ex.get("locomotive_number"))
        key = keys[0] if keys else (hint_t or ("774" if 360 <= post_dt.hour * 60 + post_dt.minute < 810 else "773"), journey_date_for(hint_t or "773", post_dt))
        meta_by_j[key].append({
            "post_dt": post_dt,
            "loco_num": loco_num,
            "loco_cls": loco_cls,
            "driver": ex.get("driver_name"),
            "platform": ex.get("platform_number"),
            "crossings": ex.get("crossings") or [],
            "priority_loss": bool(ex.get("priority_loss")),
            "favored_train": ex.get("favored_train"),
            "engine_issue": bool(ex.get("engine_issue")),
            "stated": ex.get("stated_delay_minutes"),
            "stated_ref": ex.get("stated_delay_reference"),
        })

    return obs_by_j, meta_by_j


# =========================================================================
# 3. SYNTHESIS ENGINE (JOURNEYS & EVENTS RELATIONAL BUILDER)
# =========================================================================
def pick_obs(obs, station, events):
    for want_fc in (False, True):
        for o in obs:
            if o["station"] == station and o["event"] in events and o["is_forecast"] == want_fc:
                return o
    return None


def ref_m(train, st):
    e = TT[train][st]
    return e["arr_m"] if e["arr_m"] is not None else e["dep_m"]


def nearest_station(train, target, delays):
    order = ROUTE[train]
    ti, tref = order.index(target), ref_m(train, target)
    best = None
    for st, d in delays.items():
        if st == target:
            continue
        key = (abs(ref_m(train, st) - tref), 0 if order.index(st) < ti else 1)
        if best is None or key < best[0]:
            best = (key, st, d)
    return best


def resolve_departure(train, jd, obs, metas):
    order = ROUTE[train]
    origin = order[0]
    sched_dep = TT[train][origin]["dep_m"]
    base = day0(jd)

    dep_o = pick_obs(obs, origin, {"departure"})
    delay, src = None, None

    if dep_o and not dep_o["is_forecast"]:
        delay, src = max(0.0, dep_o["delay"]), "direct_post"
    else:
        stated = next((m for m in metas if m["stated"] is not None and m["stated_ref"] == "from_origin"), None)
        if stated:
            delay, src = float(stated["stated"]), "stated_in_post"
        else:
            down = [o for st in order[1:] for o in [pick_obs(obs, st, {"arrival", "pass", "departure"})] if o]
            if down:
                d1 = down[0]["delay"]
                if d1 <= 15:
                    delay, src = 0.0, "inferred_on_time"
                else:
                    delay, src = max(0.0, d1), f"estimated_from_{down[0]['station']}"

    if delay is None:
        delay, src = 0.0, "default_on_time"

    return {"delay": delay, "source": src, "dt": base + timedelta(minutes=sched_dep + delay)}


def process_full_dataset(json_file_path="output/posts.json"):
    raw = json.loads(Path(json_file_path).read_text(encoding="utf-8"))
    seen, posts = set(), []
    for p in raw:
        k = post_key(p)
        if k not in seen and (p.get("text") or "").strip():
            seen.add(k)
            posts.append(p)
    print(f"[+] Loaded {len(posts)} unique Facebook posts", flush=True)

    extractions = extract_all(posts)
    obs_by_j, meta_by_j = build_events(posts, extractions)

    # ---------------------------------------------------------------------
    # BUILD TABLE 1: events.csv (Checkpoint-Level Granular Data)
    # ---------------------------------------------------------------------
    event_rows = []
    for (t, jd), lst in obs_by_j.items():
        if jd.weekday() == 4:
            continue
        j_id = f"{jd.isoformat()}_{t}"
        tt = TT[t]
        order = ROUTE[t]

        # Sort chronologically
        lst_sorted = sorted(lst, key=lambda o: o["actual"])

        for o in lst_sorted:
            st = o["station"]
            sched_entry = tt.get(st, {})
            sched_m = sched_entry.get("arr_m") or sched_entry.get("dep_m")
            dist_km = sched_entry.get("dist_km")

            # Match any crossings reported at this station
            matching_cross = []
            for m in meta_by_j.get((t, jd), []):
                for c in m.get("crossings", []):
                    c_st = canonical_station(c.get("station"))
                    if c_st == st or c_st is None:
                        matching_cross.append(c)

            c_trains = "; ".join(dict.fromkeys(c.get("other_train") for c in matching_cross if c.get("other_train"))) or None
            c_wait = max([c.get("wait_minutes") for c in matching_cross if c.get("wait_minutes")], default=None)

            event_rows.append({
                "journey_id": j_id,
                "journey_date": jd.isoformat(),
                "train_number": t,
                "station": st,
                "station_sequence": order.index(st) if st in order else None,
                "distance_km_from_origin": dist_km,
                "stops_remaining": (len(order) - 1 - order.index(st)) if st in order else None,
                "event_type": o["event"],
                "event_time_24h": o["actual"].strftime("%H:%M"),
                "scheduled_time_24h": mm_to_hhmm(sched_m) if sched_m else None,
                "delay_at_event_min": round(o["delay"]),
                "waiting_time_min": c_wait,
                "crossing_flag": int(bool(matching_cross or o["event"] == "halt")),
                "crossing_train": c_trains,
                "is_forecast": int(o["is_forecast"]),
                "post_time_24h": o["post_dt"].strftime("%Y-%m-%d %H:%M"),
                "post_url": o.get("url", ""),
            })

    events_df = pd.DataFrame(event_rows).sort_values(["journey_id", "distance_km_from_origin"])
    events_file = OUT / "events.csv"
    events_df.to_csv(events_file, index=False, encoding="utf-8-sig")
    print(f"\n[✓] TABLE 1 GENERATED: {events_file} ({len(events_df)} station events)")

    # ---------------------------------------------------------------------
    # BUILD TABLE 2: journeys.csv (Journey-Level Trip Master)
    # ---------------------------------------------------------------------
    journeys_dict = {}
    for (t, jd), obs in obs_by_j.items():
        if jd.weekday() == 4:  # Friday off-day
            continue
        metas = sorted(meta_by_j.get((t, jd), []), key=lambda m: m["post_dt"], reverse=True)
        journeys_dict[(t, jd)] = (obs, metas)

    dep_info = {k: resolve_departure(k[0], k[1], o, m) for k, (o, m) in journeys_dict.items()}
    dep_ctx = {jd: info for (t, jd), info in dep_info.items() if t == "773"}

    journey_rows = []
    for (train, jd), (obs, metas) in journeys_dict.items():
        tt, order = TT[train], ROUTE[train]
        origin, dest = order[0], order[-1]
        base = day0(jd)
        j_id = f"{jd.isoformat()}_{train}"

        delays = {origin: dep_info[(train, jd)]["delay"]}
        for st in order[1:]:
            o = pick_obs(obs, st, {"arrival", "pass", "halt", "departure"})
            if o and not o["is_forecast"]:
                delays[st] = o["delay"]

        # Ground truth destination arrival
        dest_o = pick_obs(obs, dest, {"arrival", "pass"})
        dest_delay_ground = max(0.0, dest_o["delay"]) if (dest_o and not dest_o["is_forecast"]) else None
        dest_time_ground = t24(dest_o["actual"]) if (dest_o and not dest_o["is_forecast"]) else None

        # Imputed destination arrival (for full sample models)
        dest_delay_imputed = dest_delay_ground
        dest_source = "direct_post"
        label_weight = 1.0

        if dest_delay_imputed is None:
            near = nearest_station(train, dest, delays)
            if near and near[0][0] <= 180:
                dest_delay_imputed = max(0.0, near[2])
                dest_source = f"estimated_from_{near[1]}"
                label_weight = 0.50
            elif train == "774":
                d773 = dep_ctx.get(jd)
                if d773 and d773["delay"] is not None and d773["delay"] >= 45 and d773["source"] != "inferred_on_time":
                    dest_delay_imputed = max(0.0, d773["delay"] - 15)
                    dest_source = "inferred_from_next_departure_rake"
                    label_weight = 0.25
            elif near:
                dest_delay_imputed = max(0.0, near[2])
                dest_source = f"estimated_from_{near[1]}"
                label_weight = 0.35

        # Shaistaganj Checkpoint
        sh_o = pick_obs(obs, "Shaistaganj", {"arrival", "pass", "halt", "departure"})
        sh_delay = max(0.0, sh_o["delay"]) if sh_o else None
        sh_time = t24(sh_o["actual"]) if sh_o else None
        sh_source = "direct_post" if sh_o else "none"

        if sh_delay is None:
            near_sh = nearest_station(train, "Shaistaganj", delays)
            if near_sh:
                sh_delay = max(0.0, near_sh[2])
                sh_time = mm_to_hhmm(tt["Shaistaganj"]["arr_m"] + int(round(sh_delay)))
                sh_source = f"estimated_from_{near_sh[1]}"

        # Actual arrival time construction
        dest_time_final = dest_time_ground or (mm_to_hhmm(tt[dest]["arr_m"] + int(round(dest_delay_imputed))) if dest_delay_imputed is not None else None)

        # Meta attributes
        crossings = [c for m in metas for c in m["crossings"]]
        locos = [m["loco_num"] for m in metas if m["loco_num"]]
        classes = [m["loco_cls"] for m in metas if m["loco_cls"]]
        drivers = [m["driver"] for m in metas if m["driver"]]
        platforms = [m["platform"] for m in metas if m["platform"]]
        priority_loss = int(any(m["priority_loss"] for m in metas))
        favored_trains = "; ".join(dict.fromkeys(m["favored_train"] for m in metas if m["favored_train"])) or None
        engine_issue = int(any(m["engine_issue"] for m in metas))

        # Traffic conflicts from Kamalapur timetable
        day_name = jd.strftime("%A")
        dep_delay_val = dep_info[(train, jd)]["delay"]
        is_parabat_running = int(day_name != "Tuesday")

        subarna_dep_clash = 0
        if train == "773" and day_name != "Friday":
            actual_dep_m = 895 + dep_delay_val
            if 915 <= actual_dep_m <= 945:  # Clashes with Subarna 15:30 departure
                subarna_dep_clash = 1

        subarna_arr_clash = 0
        if train == "774" and day_name != "Friday" and dest_delay_imputed is not None:
            actual_arr_m = 775 + dest_delay_imputed
            if abs(actual_arr_m - 760) <= 25:  # Within 25m of Subarna 12:40 arrival
                subarna_arr_clash = 1

        total_journey_time = None
        if dep_info[(train, jd)]["dt"] and dest_time_final:
            th, tm = map(int, dest_time_final.split(":"))
            dest_m = th * 60 + tm
            if train == "773" and th < 6:
                dest_m += 1440
            total_journey_time = dest_m - (sched_minutes(train, origin, "departure") + int(round(dep_delay_val)))

        journey_rows.append({
            "journey_id": j_id,
            "journey_date": jd.isoformat(),
            "journey_day": day_name,
            "train_number": train,
            "direction": "Dhaka to Sylhet" if train == "773" else "Sylhet to Dhaka",
            "origin_station": origin,
            "destination_station": dest,

            # Targets (Regression & Classification)
            "destination_delay_min_ground_truth": int(round(dest_delay_ground)) if dest_delay_ground is not None else None,
            "destination_delay_min_imputed": int(round(dest_delay_imputed)) if dest_delay_imputed is not None else None,
            "label_weight": label_weight,
            "is_destination_direct": int(dest_delay_ground is not None),
            "destination_arrival_time_24h": dest_time_final,
            "destination_source": dest_source,

            # Departure Status
            "scheduled_departure_24h": mm_to_hhmm(tt[origin]["dep_m"]),
            "departure_time_24h": t24(dep_info[(train, jd)]["dt"]),
            "departure_delay_min": int(round(dep_delay_val)),
            "departure_source": dep_info[(train, jd)]["source"],

            # Shaistaganj Status
            "scheduled_shaistaganj_arrival_24h": mm_to_hhmm(tt["Shaistaganj"]["arr_m"]),
            "shaistaganj_time_24h": sh_time,
            "shaistaganj_delay_min": int(round(sh_delay)) if sh_delay is not None else None,
            "shaistaganj_source": sh_source,

            # Performance & Root Causes
            "total_journey_time_min": total_journey_time,
            "locomotive_number": locos[0] if locos else None,
            "locomotive_class": classes[0] if classes else "Unknown",
            "is_monster_engine_3000": int(classes and classes[0] == "Class_3000_Monster"),
            "engine_trouble_reported": engine_issue,
            "train_driver": drivers[0] if drivers else None,
            "platform_number": platforms[0] if platforms else None,

            # Operational Conflict Signals
            "is_parabat_operating_today": is_parabat_running,
            "subarna_departure_slot_clash": subarna_dep_clash,
            "subarna_arrival_outer_clash": subarna_arr_clash,
            "priority_loss_flag": priority_loss,
            "favored_train": favored_trains,
            "crossing_count": len(crossings),
            "crossing_trains": "; ".join(dict.fromkeys(c.get("other_train") for c in crossings if c.get("other_train"))) or None,

            # Observation Volume
            "n_events_recorded": len(obs),
            "n_facebook_posts": len({o["url"] or o["post_dt"] for o in obs}),
        })

    journeys_df = pd.DataFrame(journey_rows).sort_values(["journey_date", "train_number"]).reset_index(drop=True)

    # Add Inbound Cascading Rake Lag Features
    lookup_delay = {(r.journey_date, r.train_number): r.destination_delay_min_imputed for r in journeys_df.itertuples()}
    g = journeys_df.groupby("train_number")["destination_delay_min_imputed"]
    journeys_df["prev_delay_same_train_min"] = g.shift(1)
    journeys_df["rolling3_delay_same_train_min"] = g.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())

    def calc_inbound(r):
        d = datetime.strptime(r.journey_date, "%Y-%m-%d").date()
        if r.train_number == "773":
            key = (d.isoformat(), "774")
            buffer_min = 120.0
        else:
            key = ((d - timedelta(days=1)).isoformat(), "773")
            buffer_min = 525.0

        inbound = lookup_delay.get(key)
        if inbound is not None and not pd.isna(inbound):
            # Explicit float conversion removes the Pylance Scalar/timedelta64 warning
            inbound_val = float(inbound)
            deficit = max(0.0, inbound_val - buffer_min)
            return inbound_val, deficit

        return None, None

    cascades = [calc_inbound(r) for r in journeys_df.itertuples()]
    journeys_df["inbound_rake_delay_min"] = [c[0] for c in cascades]
    journeys_df["inbound_turnaround_deficit_min"] = [c[1] for c in cascades]

    journeys_file = OUT / "journeys.csv"
    journeys_df.to_csv(journeys_file, index=False, encoding="utf-8-sig")
    print(f"[✓] TABLE 2 GENERATED: {journeys_file} ({len(journeys_df)} complete journeys)")

    # Also export the master ML-ready dataset
    ml_file = OUT / "kalni_train_delay_ml_dataset.csv"
    journeys_df.to_csv(ml_file, index=False, encoding="utf-8-sig")
    print(f"[✓] MASTER ML DATASET EXPORTED: {ml_file}")


if __name__ == "__main__":
    process_full_dataset("output/posts.json")