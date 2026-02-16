# PAC Roster Trainer (Streamlit)

Internal study tool to drill **name ↔ team ↔ position** with **faces + optional audio**.

This app pulls live rosters + headshots from ESPN's public JSON endpoints at runtime.

## 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # mac/linux
# .venv\Scripts\activate  # windows
pip install -r requirements.txt
```

## 2) Run
```bash
streamlit run app.py
```

## 3) Use
- Select **NFL** or **NBA**
- Pick one or more teams (defaults to **BAL** for NFL)
- Choose mode: **Quiz**, **Flash**, or **Audio**
- Your progress saves locally to `storage/progress.json` (ignored by git)
