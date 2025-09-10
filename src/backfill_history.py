#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Backfill alternativo usando nflverse (nfl_data_py).
Llena data/historical_games.csv con temporadas pasadas (finales),
mapeando a códigos de equipo consistentes con tus scripts (ARI, LAR, LV, WAS, JAX...).

No requiere SportsDataIO. Si ya existe el CSV, mergea sin duplicar.
"""

import pandas as pd
from pathlib import Path

# Temporadas a cargar (ajusta a gusto)
YEARS = list(range(2020, 2025))  # 2020–2024

OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "historical_games.csv"

# Variantes -> código estándar que usa tu modelo
TEAM_FIX = {
    # Igual ya estándar
    "ARI":"ARI","ATL":"ATL","BAL":"BAL","BUF":"BUF","CAR":"CAR","CHI":"CHI","CIN":"CIN","CLE":"CLE",
    "DAL":"DAL","DEN":"DEN","DET":"DET","GB":"GB","HOU":"HOU","IND":"IND","JAX":"JAX","KC":"KC",
    "LV":"LV","LAC":"LAC","LAR":"LAR","MIA":"MIA","MIN":"MIN","NE":"NE","NO":"NO","NYG":"NYG","NYJ":"NYJ",
    "PHI":"PHI","PIT":"PIT","SF":"SF","SEA":"SEA","TB":"TB","TEN":"TEN","WAS":"WAS",
    # Variantes históricas
    "LA":"LAR","STL":"LAR",     # Rams
    "SD":"LAC",                 # Chargers
    "OAK":"LV",                 # Raiders
    "WSH":"WAS",                # Washington
    "JAC":"JAX"                 # Jaguars
}

def map_team(x: str) -> str:
    if pd.isna(x): return ""
    x = str(x).strip().upper()
    return TEAM_FIX.get(x, x)

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
    import nfl_data_py as nfl

    # 1) Cargar schedules del nflverse
    print(f"[backfill-nflverse] Cargando YEARS={YEARS}")
    sched = nfl.import_schedules(years=YEARS)  # requiere pyarrow
    # Columnas comunes esperadas en nflverse:
    # season, week, game_type, game_id, home_team, away_team, home_score, away_score, game_date (a veces 'gameday')
    if "game_date" not in sched.columns:
        # versiones antiguas usan 'gameday'
        if "gameday" in sched.columns:
            sched = sched.rename(columns={"gameday": "game_date"})
        else:
            sched["game_date"] = pd.NaT

    # 2) Filtrar solo juegos con marcador (finalizados)
    sched = sched[(sched["home_score"].notna()) & (sched["away_score"].notna())].copy()

    # 3) Mapear equipos a códigos estándar
    sched["home"] = sched["home_team"].map(map_team)
    sched["away"] = sched["away_team"].map(map_team)

    # 4) Seleccionar columnas y normalizar tipos
    out = pd.DataFrame({
        "season": sched["season"].astype("Int64"),
        "week":   pd.to_numeric(sched["week"], errors="coerce").astype("Int64"),
        "home":   sched["home"],
        "away":   sched["away"],
        "home_score": pd.to_numeric(sched["home_score"], errors="coerce").astype("Int64"),
        "away_score": pd.to_numeric(sched["away_score"], errors="coerce").astype("Int64"),
        "game_date": pd.to_datetime(sched["game_date"], errors="coerce").astype(str),
        "game_id":  sched["game_id"].fillna("").astype(str),
        "closing_spread_home": ""  # reservado por si luego cargas closing lines
    })

    # 5) Merge con existente (idempotente)
    prev = load_existing()
    if "game_id" in prev.columns and prev["game_id"].notna().any():
        key_cols = ["game_id"]
    else:
        key_cols = ["season","week","home","away"]

    merged = pd.concat([prev, out], ignore_index=True).drop_duplicates(subset=key_cols, keep="last")
    merged = merged.sort_values(["season","week","game_date","game_id"], na_position="last")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)

    print(f"[backfill-nflverse] Guardado {OUT_PATH} | previas={len(prev)} nuevas={len(out)} total={len(merged)}")

if __name__ == "__main__":
    main()
