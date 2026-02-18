import json
import os
import random
import time
import io
import urllib.parse
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
import streamlit as st
from PIL import Image


APP_TITLE = "PAC Roster Trainer (NFL + NBA)"
PROGRESS_PATH = "storage/progress.json"

# ESPN "site" API base
ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

# ---------------------------
# Storage (local progress)
# ---------------------------
def ensure_storage():
    os.makedirs(os.path.dirname(PROGRESS_PATH), exist_ok=True)
    if not os.path.exists(PROGRESS_PATH):
        with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f)

def load_progress() -> Dict:
    ensure_storage()
    with open(PROGRESS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_progress(progress: Dict) -> None:
    ensure_storage()
    with open(PROGRESS_PATH, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)

def normalize(s: str) -> str:
    return "".join(ch for ch in (s or "").strip().upper() if ch.isalnum())

def is_correct(user_team: str, user_pos: str, true_team: str, true_pos: str) -> bool:
    return normalize(user_team) == normalize(true_team) and normalize(user_pos) == normalize(true_pos)

def now_ts() -> int:
    return int(time.time())

# ---------------------------
# ESPN API helpers
# ---------------------------
@st.cache_data(ttl=60 * 60)  # cache 1 hour
def fetch_teams(league: str) -> List[dict]:
    """
    league: "nfl" or "nba"
    Returns list of team dicts with id, abbreviation, displayName, logos.
    """
    url = f"{ESPN_BASE}/football/nfl/teams" if league == "nfl" else f"{ESPN_BASE}/basketball/nba/teams"
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    teams = []
    # structure: sports -> leagues -> teams
    for entry in data.get("sports", [])[0].get("leagues", [])[0].get("teams", []):
        t = entry.get("team", {})
        teams.append({
            "id": str(t.get("id")),
            "abbrev": t.get("abbreviation"),
            "name": t.get("displayName"),
            "short": t.get("shortDisplayName"),
            "logo": (t.get("logos", [{}])[0].get("href") if t.get("logos") else None),
        })
    return teams

@st.cache_data(ttl=30 * 60)  # cache 30 minutes
def fetch_roster(league: str, team_id: str) -> List[dict]:
    """
    Returns list of players:
      name, team_abbrev, position_abbrev, headshot_url
    """
    if league == "nfl":
        url = f"{ESPN_BASE}/football/nfl/teams/{team_id}/roster"
    else:
        url = f"{ESPN_BASE}/basketball/nba/teams/{team_id}/roster"

    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()

    # Team abbreviation is available at: data["team"]["abbreviation"] usually
    team_abbrev = (data.get("team", {}) or {}).get("abbreviation")

    players = []
    # roster entries in "athletes" groups (e.g., offense/defense/ST or guards/forwards/centers)
    for group in data.get("athletes", []):
        for a in group.get("items", []):
            # name
            name = a.get("fullName") or a.get("displayName")
            # position
            pos = (a.get("position") or {}).get("abbreviation") or (a.get("position") or {}).get("name")
            # headshot
            hs = None
            if a.get("headshot") and isinstance(a["headshot"], dict):
                hs = a["headshot"].get("href")
            # some entries might not have headshots
            players.append({
                "league": league.upper(),
                "team": team_abbrev,
                "position": pos,
                "name": name,
                "image_url": hs,
            })

    # fallback: sometimes ESPN uses "athletes" differently; try "team" -> "athletes"
    if not players:
        roster = (data.get("team", {}) or {}).get("athletes", [])
        for a in roster:
            name = a.get("fullName") or a.get("displayName")
            pos = (a.get("position") or {}).get("abbreviation") or (a.get("position") or {}).get("name")
            hs = (a.get("headshot") or {}).get("href") if isinstance(a.get("headshot"), dict) else None
            players.append({"league": league.upper(), "team": team_abbrev, "position": pos, "name": name, "image_url": hs})

    # clean
    players = [p for p in players if p["name"] and p["team"] and p["position"]]
    return players

@st.cache_data(ttl=7 * 24 * 60 * 60)  # cache 7 days
def wikipedia_thumbnail_url(player_name: str, league: str) -> Optional[str]:
    """
    Try to get a usable headshot-style thumbnail from Wikipedia.
    This avoids hotlink/CDN blocking issues from ESPN images.
    """
    if not player_name:
        return None

    # Add a tiny bit of disambiguation to improve hit rate
    query = player_name
    if league.lower() == "nfl":
        query += " NFL"
    elif league.lower() == "nba":
        query += " NBA"

    try:
        # 1) Search for the best matching title
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 1,
        }
        r = requests.get(search_url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        results = (data.get("query", {}) or {}).get("search", [])
        if not results:
            return None

        title = results[0].get("title")
        if not title:
            return None

        # 2) Get the REST summary (includes thumbnail if available)
        encoded_title = urllib.parse.quote(title.replace(" ", "_"))
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}"
        r2 = requests.get(summary_url, timeout=20)
        if r2.status_code != 200:
            return None
        summary = r2.json()

        thumb = (summary.get("thumbnail") or {}).get("source")
        if thumb:
            return thumb

        # Sometimes Wikipedia uses "originalimage" instead
        orig = (summary.get("originalimage") or {}).get("source")
        return orig

    except Exception:
        return None


def best_headshot_bytes(card: dict) -> Optional[bytes]:
    """
    Fallback chain (prioritizes actually-decodable images):
      1) Wikipedia thumbnail (usually JPG/PNG and reliable)
      2) ESPN headshot (skip if AVIF/HEIC/HEIF detected)
    """

    # 1) Wikipedia first (more reliable for decoding/rendering)
    wiki_url = wikipedia_thumbnail_url(card.get("name", ""), card.get("league", ""))
    if wiki_url:
        img = fetch_image_bytes(wiki_url)
        if isinstance(img, (bytes, bytearray)) and len(img) > 200:
            return bytes(img)

    # 2) ESPN second
    espn_url = card.get("image_url")
    if espn_url:
        img2 = fetch_image_bytes(espn_url)
        if isinstance(img2, (bytes, bytearray)) and len(img2) > 200:
            return bytes(img2)

    return None


# ---------------------------
# Training logic
# ---------------------------
@dataclass
class CardResult:
    correct: bool
    response_time_ms: int

def pick_weighted_card(pool: List[dict], stats: Dict[str, Dict]) -> dict:
    """
    Weighted sampling: prioritize cards with misses / low streak / unseen.
    """
    weights = []
    for p in pool:
        key = f'{p["league"]}:{p["team"]}:{p["name"]}'
        s = stats.get(key, {})
        seen = s.get("seen", 0)
        misses = s.get("misses", 0)
        streak = s.get("streak", 0)
        w = 1.0
        if seen == 0:
            w += 3.0
        w += float(misses) * 2.0
        w += max(0.0, 5.0 - float(streak))
        weights.append(w)

    total = sum(weights)
    r = random.uniform(0, total)
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if upto >= r:
            return pool[i]
    return random.choice(pool)

def record_result(progress: Dict, username: str, card: dict, result: CardResult):
    user = progress.setdefault(username, {})
    stats = user.setdefault("stats", {})
    key = f'{card["league"]}:{card["team"]}:{card["name"]}'
    s = stats.setdefault(key, {"seen": 0, "correct": 0, "misses": 0, "streak": 0, "avg_ms": None})

    s["seen"] += 1
    if result.correct:
        s["correct"] += 1
        s["streak"] = s.get("streak", 0) + 1
    else:
        s["misses"] += 1
        s["streak"] = 0

    prev = s.get("avg_ms")
    if prev is None:
        s["avg_ms"] = result.response_time_ms
    else:
        s["avg_ms"] = int(prev * 0.8 + result.response_time_ms * 0.2)

    user["last_seen_ts"] = now_ts()

def summarize(pool: List[dict], stats: Dict[str, Dict]) -> tuple[int, int, float]:
    total = len(pool)
    seen = 0
    correct = 0
    attempts = 0
    for p in pool:
        key = f'{p["league"]}:{p["team"]}:{p["name"]}'
        s = stats.get(key, {})
        if s.get("seen", 0) > 0:
            seen += 1
        correct += s.get("correct", 0)
        attempts += s.get("seen", 0)
    acc = (correct / attempts) if attempts else 0.0
    return total, seen, acc

# ---------------------------
# UI helpers
# ---------------------------
def tts_button(text: str, label: str = "🔊 Speak"):
    safe = (text or "").replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <button style="padding:8px 12px;border-radius:10px;border:1px solid #444;background:#111;color:#fff;cursor:pointer;"
        onclick='(function(){{
            const u = new SpeechSynthesisUtterance("{safe}");
            u.rate = 1.0;
            u.pitch = 1.0;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(u);
        }})()'>
        {label}
    </button>
    """
    st.components.v1.html(html, height=50)

def fetch_image_bytes(url: str) -> Optional[bytes]:
    """
    Fetch image server-side. Returns bytes or None.
    Adds format-detection so we can skip AVIF/HEIC/HEIF which often fail on Streamlit Cloud.
    """
    if not url:
        return None

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.espn.com/",
        # Strongly prefer formats we can decode
        "Accept": "image/jpeg,image/png,image/webp;q=0.9,image/*;q=0.8,*/*;q=0.1",
    }

    try:
        r = requests.get(url, headers=headers, timeout=20, allow_redirects=True)
        if r.status_code != 200:
            return None

        data = r.content
        if not data or len(data) < 200:
            return None

        # ---------
        # Detect AVIF / HEIC / HEIF by common ISO BMFF "ftyp" brands in header.
        # Many CDNs return AVIF even when you ask for jpg/png.
        # Pillow frequently can't decode AVIF/HEIC on Streamlit Cloud.
        # ---------
        head = data[:64]
        if b"ftypavif" in head or b"ftypavis" in head:
            return None
        if b"ftypheic" in head or b"ftypheix" in head or b"ftypheif" in head or b"ftypmif1" in head:
            return None

        return data

    except Exception:
        return None

def show_image(card: dict):
    """
    Fetch image bytes, decode with Pillow, convert to PNG, then render.
    This fixes cases where the source returns AVIF/odd formats that Streamlit can't display directly.
    """
    img = best_headshot_bytes(card)

    # DEBUG (keep for now)
    st.caption(f"Image bytes received: {0 if img is None else len(img)}")

    if not isinstance(img, (bytes, bytearray)) or len(img) < 200:
        st.caption("No headshot available (blocked or not found).")
        return

    try:
        # Decode whatever format we got
        im = Image.open(io.BytesIO(bytes(img)))
        im = im.convert("RGB")  # normalize

        # Convert to PNG in-memory
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        buf.seek(0)

        st.image(buf, use_container_width=True)

    except Exception:
        st.caption("Headshot failed to render (decode/convert error).")


def init_state():
    st.session_state.setdefault("mode", "Quiz")
    st.session_state.setdefault("current_card", None)
    st.session_state.setdefault("card_start_ts", None)
    st.session_state.setdefault("reveal", False)
    st.session_state.setdefault("last_result", None)

    # NEW: input-clearing mechanism
    st.session_state.setdefault("clear_inputs", False)

    # If a previous action requested clearing, do it BEFORE widgets render
    if st.session_state.clear_inputs:
        st.session_state["quiz_team"] = ""
        st.session_state["quiz_pos"] = ""
        st.session_state["audio_team"] = ""
        st.session_state["audio_pos"] = ""
        st.session_state.clear_inputs = False


def set_next_card(pool: List[dict], user_stats: Dict[str, Dict]):
    st.session_state.current_card = pick_weighted_card(pool, user_stats)
    st.session_state.card_start_ts = now_ts()
    st.session_state.reveal = False
    st.session_state.last_result = None

# ---------------------------
# App
# ---------------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Internal recall trainer: face + name + team + position. Uses ESPN public JSON endpoints for rosters/headshots.")

init_state()
progress = load_progress()

# Sidebar setup
with st.sidebar:
    st.header("Setup")

    username = st.text_input("User name (saves progress locally)", value="mike").strip() or "user"

    league_label = st.selectbox("League", ["NFL", "NBA"])
    league = "nfl" if league_label == "NFL" else "nba"

    st.divider()
    st.header("Team selection")
    try:
        teams = fetch_teams(league)
    except Exception as e:
        st.error(f"Failed to load {league_label} teams from ESPN API: {e}")
        st.stop()

    team_options = {f'{t["abbrev"]} — {t["name"]}': t for t in teams if t.get("id") and t.get("abbrev")}
    default_keys = []
    # Default you into Ravens for NFL
    if league == "nfl":
        for k, t in team_options.items():
            if t["abbrev"] == "BAL":
                default_keys = [k]
                break

    selected = st.multiselect("Teams", options=list(team_options.keys()), default=default_keys)

    st.divider()
    st.header("Mode")
    mode = st.radio("Choose mode", options=["Quiz", "Flash", "Audio"], index=["Quiz","Flash","Audio"].index(st.session_state.mode))
    st.session_state.mode = mode
    flash_seconds = st.slider("Flash: seconds per card", 1, 6, 2)

    st.divider()
    include_no_headshots = st.checkbox("Include players without headshots", value=False)

    st.divider()
    if st.button("Reset my progress", type="secondary"):
        if username in progress:
            del progress[username]
            save_progress(progress)
        st.success("Progress cleared for this user.")

# Build pool
if not selected:
    st.warning("Select at least one team.")
    st.stop()

pool: List[dict] = []
for key in selected:
    team = team_options[key]
    try:
        roster = fetch_roster(league, team["id"])
        for p in roster:
            # enforce team abbrev fallback
            p["team"] = p["team"] or team["abbrev"]
        pool.extend(roster)
    except Exception as e:
        st.error(f"Failed to load roster for {team['abbrev']}: {e}")

# Filter out no-headshot if desired
if not include_no_headshots:
    pool = [p for p in pool if p.get("image_url")]

# De-dupe
seen_keys = set()
deduped = []
for p in pool:
    k = f'{p["league"]}:{p["team"]}:{p["name"]}:{p["position"]}'
    if k not in seen_keys:
        seen_keys.add(k)
        deduped.append(p)
pool = deduped

if not pool:
    st.warning("No players loaded (or all headshots filtered out). Try enabling 'Include players without headshots'.")
    st.stop()

user = progress.setdefault(username, {})
user_stats = user.setdefault("stats", {})

# Metrics
total, seen, acc = summarize(pool, user_stats)
m1, m2, m3 = st.columns(3)
m1.metric("Cards in pool", total)
m2.metric("Seen at least once", seen)
m3.metric("Accuracy", f"{acc*100:.1f}%")

st.divider()

# Initialize card
if st.session_state.current_card is None:
    set_next_card(pool, user_stats)

card = st.session_state.current_card

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("Card")
    show_image(card)


    if st.session_state.mode != "Flash":
        if st.session_state.reveal:
            st.markdown(f'### **{card["name"]}**')
            st.write(f'**Team:** {card["team"]}  |  **Position:** {card["position"]}')
        else:
            st.markdown("### (Answer first)")

with right:
    st.subheader("Recall")

    if st.session_state.mode == "Flash":
        st.markdown(f'### **{card["name"]}**')
        st.write(f'**Team:** {card["team"]}  |  **Position:** {card["position"]}')
        tts_button(card["name"], label="🔊 Speak name")

        c1, c2 = st.columns(2)
        if c1.button("Next now", type="primary"):
            set_next_card(pool, user_stats)
            st.rerun()

        st.caption(f"Auto-advances every {flash_seconds}s (training tape).")
        time.sleep(flash_seconds)
        set_next_card(pool, user_stats)
        st.rerun()

    elif st.session_state.mode == "Audio":
        st.write("Press **Speak**, then answer from audio only (no name shown until reveal).")
        tts_button(card["name"], label="🔊 Speak")

        user_team = st.text_input("Team (e.g., BAL)", key="audio_team")
        user_pos = st.text_input("Position (e.g., S / LT / EDGE)", key="audio_pos")

        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("Reveal", type="secondary"):
            st.session_state.reveal = True

        if c2.button("Submit", type="primary"):
            rt_ms = max(0, (now_ts() - st.session_state.card_start_ts) * 1000)
            correct = is_correct(user_team, user_pos, card["team"], card["position"])
            st.session_state.reveal = True
            st.session_state.last_result = "correct" if correct else "miss"
            record_result(progress, username, card, CardResult(correct=correct, response_time_ms=rt_ms))
            save_progress(progress)

        if c3.button("Next"):
            st.session_state.clear_inputs = True
            set_next_card(pool, user_stats)
            st.rerun()


    else:
        st.write("Look at the face and answer **team + position** in under 2 seconds.")
        user_team = st.text_input("Team (e.g., BAL)", key="quiz_team")
        user_pos = st.text_input("Position (e.g., CB / LG / EDGE)", key="quiz_pos")

        c1, c2, c3 = st.columns([1, 1, 1])
        if c1.button("Reveal", type="secondary"):
            st.session_state.reveal = True

        if c2.button("Submit", type="primary"):
            rt_ms = max(0, (now_ts() - st.session_state.card_start_ts) * 1000)
            correct = is_correct(user_team, user_pos, card["team"], card["position"])
            st.session_state.reveal = True
            st.session_state.last_result = "correct" if correct else "miss"
            record_result(progress, username, card, CardResult(correct=correct, response_time_ms=rt_ms))
            save_progress(progress)

        if c3.button("Next"):
            st.session_state.clear_inputs = True
            set_next_card(pool, user_stats)
            st.rerun()


# Feedback
if st.session_state.last_result:
    if st.session_state.last_result == "correct":
        st.success("Correct.")
    else:
        st.error("Miss. Repeat it 3x out loud, then hit Next.")

with st.expander("Show my weakest cards (this pool)"):
    rows = []
    for p in pool:
        key = f'{p["league"]}:{p["team"]}:{p["name"]}'
        s = user_stats.get(key, {})
        seen_i = s.get("seen", 0)
        misses = s.get("misses", 0)
        acc_i = (1 - (misses / seen_i)) if seen_i > 0 else 0.0
        rows.append({
            "name": p["name"],
            "team": p["team"],
            "pos": p["position"],
            "seen": seen_i,
            "misses": misses,
            "accuracy_%": round(acc_i * 100, 1),
            "avg_ms": s.get("avg_ms"),
        })
    rows = sorted(rows, key=lambda x: (x["accuracy_%"], -x["seen"]))[:40]
    st.dataframe(rows, use_container_width=True)
