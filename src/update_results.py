#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Actualiza data/historical_games.csv con los resultados FINAL de la NFL
usando SportsDataIO. Se ejecuta seguro múltiples veces: es idempotente.

Columnas de salida:
  season, week, home, away, home_score, away_score, game_date, game_id, closing_spread_home
(closing_spread_home se deja vacío; tu modelo no la usa para entrenar)
"""

import os
import sys
import json
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime

BASE = "https://api.sportsdata.io/v3/nfl/scores/json/"
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"

def sdi_get(endpoint: str, key: str):
    url = BASE + endpoint.lstrip("/")
    headers = {"Ocp-Apim-Subscription-Key": key}
    r = requests.get(url, headers=headers, timeout=40)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        raise RuntimeError(f"Respuesta no-JSON para {endpoint}: {r.text[:200]}")

def get_current_season(key: str) -> int:
    """
    Preferimos /CurrentSeason; si falla, intentamos /Timeframes/current.
    """
    try:
        cur = sdi_get("CurrentSeason", key)
        return int(cur)
    except Exception:
        tf = sdi_get("Timeframes/current", key)
        # 'Timeframes' devuelve lista con objetos que traen Season, SeasonType, etc.
        if isinstance(tf, list) and len(tf) > 0 and "Season" in tf[0]:
            return int(tf[0]["Season"])
        raise

def get_last_completed_week(key: str) -> int:
    """
    Devuelve el último 'Regular Season' week completado.
    Si 0 (pretemporada), devolvemos 0 y el script no añadirá nada.
    """
    wk = sdi_get("LastCompletedWeek", key)
    try:
        return int(wk)
    except Exception:
        # Fallback: CurrentWeek-1
        cwk = int(sdi_get("CurrentWeek", key))
        return max(0, cwk - 1)

def fetch_week_finals(season: int, week: int, key: str) -> list:
    """
    Usa el endpoint 'ScoresByWeekFinal/{season}/{week}' para traer sólo FINAL.
    """
    ep = f"ScoresByWeekFinal/{season}/{week}"
    games = sdi_get(ep, key)
    return games if isinstance(games, list) else []

def normalize_rows(glist: list, season: int, week: int) -> list:
    rows = []
    for g in glist:
        # Campos típicos en Scores endpoints
        home = g.get("HomeTeam")
        away = g.get("AwayTeam")
        # Scores
        hs = g.get("HomeScore")
        as_ = g.get("AwayScore")
        # Fecha (ISO)
        dt = g.get("Date") or g.get("DateTime") or g.get("Day")
        # ID único
        gid = g.get("GameKey") or g.get("GameID") or f"{season}-{week}-{home}-{away}"
        # Status
        status = (g.get("Status") or "").lower()
        is_final = ("final" in status) or g.get("IsOver") is True

        if not (home and away):
            continue
        if hs is None or as_ is None:
            # por si el feed no incluye aún los puntos, aunque sea "Final"
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
            "closing_spread_home": ""  # no usado en tu training
        })
    return rows

def load_existing() -> pd.DataFrame:
    if OUT_PATH.exists():
        try:
            return pd.read_csv(OUT_PATH)
        except Exception:
            # Recuperación simple si hubiera formato viejo
            return pd.read_csv(OUT_PATH, on_bad_lines="skip")
    # DataFrame vacío con columnas esperadas
    return pd.DataFrame(columns=[
        "season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"
    ])

def main():
    key = os.environ.get("SPORTSDATAIO_KEY")
    if not key:
        print("ERROR: Falta SPORTSDATAIO_KEY en el entorno.", file=sys.stderr)
        sys.exit(1)

    df_old = load_existing()
    have_cols = set(df_old.columns)
    need_cols = {"season","week","home","away","home_score","away_score","game_date","game_id","closing_spread_home"}
    # Asegura columnas
    for c in (need_cols - have_cols):
        df_old[c] = "" if c in {"game_date","closing_spread_home","game_id"} else None

    # Temporada y última semana completada
    season = get_current_season(key)
    last_wk = get_last_completed_week(key)

    print(f"[update_results] season={season} last_completed_week={last_wk}")

    if last_wk <= 0:
        print("[update_results] No hay semanas regulares finalizadas aún. Nada que actualizar.")
        # Aún así, guardamos archivo si no existía
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.sort_values(["season","week","game_date","game_id"], na_position="last").to_csv(OUT_PATH, index=False)
        return

    all_rows = []
    # Trae TODO desde la semana 1 hasta la última completada (por si alguna corrida se perdió)
    for wk in range(1, last_wk + 1):
        try:
            games = fetch_week_finals(season, wk, key)
        except requests.HTTPError as e:
            print(f"[warn] fallo ScoresByWeekFinal/{season}/{wk}: {e}")
            time.sleep(0.5)
            continue
        rows = normalize_rows(games, season, wk)
        all_rows.extend(rows)

    df_new = pd.DataFrame(all_rows)
    if df_new.empty:
        print("[update_results] No llegaron partidos finalizados. Nada que mergear.")
        # Guardar estado actual si no existía
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_old.sort_values(["season","week","game_date","game_id"], na_position="last").to_csv(OUT_PATH, index=False)
        return

    # Merge idempotente
    if "game_id" in df_old.columns and df_old["game_id"].notna().any():
        key_cols = ["game_id"]
    else:
        key_cols = ["season","week","home","away"]

    before = len(df_old)
    df_out = pd.concat([df_old, df_new], ignore_index=True)
    df_out = df_out.drop_duplicates(subset=key_cols, keep="last")

    # Orden y tipos
    for c in ["season","week","home_score","away_score"]:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("Int64")

    df_out = df_out.sort_values(["season","week","game_date","game_id"], na_position="last")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(OUT_PATH, index=False)

    added = len(df_out) - before
    print(f"[update_results] Guardado {OUT_PATH} | filas previas={before} nuevas={len(df_new)} añadidas={added}")

if __name__ == "__main__":
    main()
