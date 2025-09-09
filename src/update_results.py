#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actualiza data/historical_games.csv con resultados FINAL NFL usando SportsDataIO.
Robusto ante planes sin acceso a ciertos endpoints: hace fallback y no rompe el job.

Salida: CSV con columnas
  season, week, home, away, home_score, away_score, game_date, game_id, closing_spread_home
"""

import os
import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path

BASE = "https://api.sportsdata.io/v3/nfl/scores/json/"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"

def safe_get(endpoint: str, key: str):
    """GET con diagnóstico: devuelve (ok, data|None, status|None, text|None)."""
    url = BASE + endpoint.lstrip("/")
    headers = {"Ocp-Apim-Subscription-Key": key}
    try:
        r = requests.get(url, headers=headers, timeout=40)
        status = r.status_code
        try:
            data = r.json()
            txt = None
        except Exception:
            data = None
            txt = r.text[:300]
        ok = (200 <= status < 300)
        if not ok:
            print(f"[safe_get] {endpoint} -> HTTP {status} | body_snippet={txt}")
        else:
            # Algunos planes devuelven 200 con un objeto error
            if isinstance(data, dict) and data.get("Message"):
                print(f"[safe_get] {endpoint} -> logical error: {data.get('Message')}")
                return False, None, status, data.get("Message")
        return ok, data, status, txt
    except requests.RequestException as e:
        print(f"[safe_get] EXC {endpoint}: {e}")
        return False, None, None, str(e)

def get_current_season(key: str) -> int | None:
    ok, data, _, _ = safe_get("CurrentSeason", key)
    if ok:
        try:
            return int(data)
        except Exception:
            pass
    # Fallback a Timeframes/current
    ok, data, _, _ = safe_get("Timeframes/current", key)
    if ok and isinstance(data, list) and data:
        try:
            return int(data[0].get("Season"))
        except Exception:
            return None
    return None

def get_last_completed_week(key: str) -> int:
    ok, data, _, _ = safe_get("LastCompletedWeek", key)
    if ok:
        try:
            return int(data)
        except Exception:
            pass
    # Fallback a CurrentWeek-1
    ok, data, _, _ = safe_get("CurrentWeek", key)
    if ok:
        try:
            cw = int(data)
            return max(0, cw - 1)
        except Exception:
            pass
    return 0

def fetch_week_any(season: int, week: int, key: str) -> list:
    """
    Intenta primero ScoresByWeekFinal; si falla, usa ScoresByWeek y filtra Final.
    """
    ep_final = f"ScoresByWeekFinal/{season}/{week}"
    ok, data, status, _ = safe_get(ep_final, key)
    if ok and isinstance(data, list):
        print(f"[fetch_week_any] {ep_final} ok ({len(data)} juegos)")
        return data

    print(f"[fetch_week_any] fallback ScoresByWeek por {ep_final} status={status}")
    ep_any = f"ScoresByWeek/{season}/{week}"
    ok, data, _, _ = safe_get(ep_any, key)
    if ok and isinstance(data, list):
        finals = []
        for g in data:
            status_txt = (g.get("Status") or "").lower()
            is_over = g.get("IsOver") is True
            if "final" in status_txt or is_over:
                finals.append(g)
        print(f"[fetch_week_any] {ep_any} ok -> finales={len(finals)} de {len(data)}")
        return finals
    print(f"[fetch_week_any] no se pudo obtener week {week}")
    return []

def normalize_rows(glist: list, season: int, week: int) -> list:
    rows = []
    for g in glist:
        home = g.get("HomeTeam")
        away = g.get("AwayTeam")
        hs = g.get("HomeScore")
        as_ = g.get("AwayScore")
        dt = g.get("Date") or g.get("DateTime") or g.get("Day")
        gid = g.get("GameKey") or g.get("GameID") or f"{season}-{week}-{home}-{away}"
        status = (g.get("Status") or "").lower()
        is_final = ("final" in status) or g.get("IsOver") is True

        if not (home and away):
            continue
        if hs is None or as_ is None:
            continue
        if not is_final:
            continue

        rows.append({
            "season": int(season),
            "week": int(week),
            "home": str(home).upper(),
            "away": str(away).upper(),
            "home_score": int(hs),
            "away_score": int(as_),
            "game_date": str(dt) if dt else "",
            "game_id": str(gid),
            "closing_spread_home": ""
        })
    return rows

def load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        try:
            return pd.read_csv(OUT_PATH)
        except Exception:
            return pd.read_csv(OUT_PATH, on_bad_lines="skip")
    return pd.DataFrame(columns=[
        "season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"
    ])

def main():
    key = os.environ.get("SPORTSDATAIO_KEY")
    if not key:
        print("ERROR: Falta SPORTSDATAIO_KEY (secret no presente).", file=sys.stderr)
        sys.exit(1)

    df_old = load_existing()
    need_cols = {"season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"}
    for c in (need_cols - set(df_old.columns)):
        df_old[c] = "" if c in {"game_date","closing_spread_home","game_id"} else None

    season = get_current_season(key)
    if not season:
        print("[update_results] No pude leer CurrentSeason/Timeframes. ¿Plan/endpoint restringido? Sigo sin actualizar.")
        # Guardamos el archivo existente para asegurar que data/ existe
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.to_csv(OUT_PATH, index=False)
        return  # no fallar el job

    last_wk = get_last_completed_week(key)
    print(f"[update_results] season={season} last_completed_week={last_wk}")

    if last_wk <= 0:
        print("[update_results] No hay semanas regulares finalizadas aún. Nada que actualizar.")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.sort_values(["season","week","game_date","game_id"], na_position="last").to_csv(OUT_PATH, index=False)
        return

    all_rows = []
    for wk in range(1, last_wk + 1):
        games = fetch_week_any(season, wk, key)
        rows = normalize_rows(games, season, wk)
        all_rows.extend(rows)
        time.sleep(0.2)  # cuida rate-limit

    df_new = pd.DataFrame(all_rows)
    if df_new.empty:
        print("[update_results] No llegaron partidos finalizados. Nada que mergear.")
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.sort_values(["season","week","game_date","game_id"], na_position="last").to_csv(OUT_PATH, index=False)
        return

    key_cols = ["game_id"] if df_old["game_id"].notna().any() else ["season","week","home","away"]
    before = len(df_old)
    df_out = pd.concat([df_old, df_new], ignore_index=True)
    df_out = df_out.drop_duplicates(subset=key_cols, keep="last")

    for c in ["season","week","home_score","away_score"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("Int64")

    df_out = df_out.sort_values(["season","week","game_date","game_id"], na_position="last")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    added = len(df_out) - before
    print(f"[update_results] Guardado {OUT_PATH} | filas previas={before} nuevas_en_feed={len(df_new)} añadidas_al_merged={added}")

if __name__ == "__main__":
    main()
