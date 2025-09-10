#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Descarga resultados FINAL de temporadas previas (p.ej. últimas 3-5)
y los fusiona en data/historical_games.csv para que Elo/RPD tengan señal real.

Usa endpoints de SportsDataIO con fallback robusto.
"""

import os
import sys
import time
import requests
import pandas as pd
from pathlib import Path

BASE = "https://api.sportsdata.io/v3/nfl/scores/json/"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"

# Ajusta las temporadas que quieras backfillear (ej. 2022–2024)
YEARS = [2022, 2023, 2024]

def safe_get(endpoint: str, key: str):
    url = BASE + endpoint.lstrip("/")
    headers = {"Ocp-Apim-Subscription-Key": key}
    try:
        r = requests.get(url, headers=headers, timeout=40)
        status = r.status_code
        try:
            data = r.json()
            txt = None
        except Exception:
            data, txt = None, r.text[:300]
        ok = 200 <= status < 300
        if not ok:
            print(f"[safe_get] {endpoint} -> HTTP {status} | {txt}")
        else:
            if isinstance(data, dict) and data.get("Message"):
                print(f"[safe_get] {endpoint} -> logical error: {data.get('Message')}")
                return False, None
        return ok, data
    except requests.RequestException as e:
        print(f"[safe_get] EXC {endpoint}: {e}")
        return False, None

def week_games_final(season: int, week: int, key: str) -> list:
    ep = f"ScoresByWeekFinal/{season}/{week}"
    ok, data = safe_get(ep, key)
    if ok and isinstance(data, list):
        return data
    # fallback
    ep2 = f"ScoresByWeek/{season}/{week}"
    ok, data = safe_get(ep2, key)
    if ok and isinstance(data, list):
        finals = []
        for g in data:
            status_txt = (g.get("Status") or "").lower()
            is_over = g.get("IsOver") is True
            if "final" in status_txt or is_over:
                finals.append(g)
        return finals
    return []

def normalize_rows(glist: list, season: int, week: int) -> list:
    rows = []
    for g in glist:
        home = (g.get("HomeTeam") or "").upper()
        away = (g.get("AwayTeam") or "").upper()
        hs = g.get("HomeScore")
        as_ = g.get("AwayScore")
        dt = g.get("Date") or g.get("DateTime") or g.get("Day")
        gid = g.get("GameKey") or g.get("GameID") or f"{season}-{week}-{home}-{away}"
        status = (g.get("Status") or "").lower()
        is_final = ("final" in status) or g.get("IsOver") is True
        if not (home and away) or hs is None or as_ is None or not is_final:
            continue
        rows.append({
            "season": int(season),
            "week": int(week),
            "home": home,
            "away": away,
            "home_score": int(hs),
            "away_score": int(as_),
            "game_date": str(dt) if dt else "",
            "game_id": str(gid),
            "closing_spread_home": ""
        })
    return rows

def load_existing(path=OUT_PATH) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.read_csv(path, on_bad_lines="skip")
    return pd.DataFrame(columns=[
        "season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"
    ])

def main():
    key = os.environ.get("SPORTSDATAIO_KEY")
    if not key:
        print("ERROR: falta SPORTSDATAIO_KEY", file=sys.stderr)
        sys.exit(1)

    df_old = load_existing()
    need_cols = {"season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"}
    for c in (need_cols - set(df_old.columns)):
        df_old[c] = "" if c in {"game_date","game_id","closing_spread_home"} else None

    all_rows = []
    for season in YEARS:
        print(f"[backfill] season={season}")
        # Weeks: 1..22 (18 RS + 4 Post aprox). Nos quedamos solo con lo que venga FINAL.
        for wk in range(1, 23):
            g = week_games_final(season, wk, key)
            if not g:
                # si no hay nada en varias semanas seguidas, rompemos
                # pero mantenemos ciclo corto
                pass
            rows = normalize_rows(g, season, wk)
            if rows:
                print(f"  wk={wk}: {len(rows)} finales")
            all_rows.extend(rows)
            time.sleep(0.15)

    df_new = pd.DataFrame(all_rows)
    if df_new.empty:
        print("[backfill] No se obtuvieron juegos. ¿Plan restringido?")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.to_csv(OUT_PATH, index=False)
        return

    key_cols = ["game_id"] if df_old["game_id"].notna().any() else ["season","week","home","away"]
    before = len(df_old)
    df_out = pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")

    for c in ["season","week","home_score","away_score"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("Int64")
    df_out = df_out.sort_values(["season","week","game_date","game_id"], na_position="last")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    print(f"[backfill] Guardado {OUT_PATH} | previas={before} nuevas={len(df_new)} total={len(df_out)}")

if __name__ == "__main__":
    main()
