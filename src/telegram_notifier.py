import os
import requests
import pandas as pd
import numpy as np
import yaml
from sklearn.linear_model import Ridge
from pathlib import Path

from elo import update_elo
from features import rolling_point_diff

# ------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------
CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / 'config.yaml'))

TEAM_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS"
}

# ------------------------------------------------------------------------------
# Datos históricos
# ------------------------------------------------------------------------------
def load_hist():
    p = Path(__file__).resolve().parents[1] / 'data' / 'historical_games.csv'
    if p.exists():
        return pd.read_csv(p)
    # Semilla mínima
    return pd.DataFrame([
        {'season': 2024, 'week': 1, 'home': 'KC',  'away': 'DET', 'home_score': 20, 'away_score': 21, 'game_date':'2024-09-07'},
        {'season': 2024, 'week': 1, 'home': 'NYJ', 'away': 'BUF', 'home_score': 22, 'away_score': 16, 'game_date':'2024-09-08'},
        {'season': 2024, 'week': 2, 'home': 'KC',  'away': 'CIN', 'home_score': 27, 'away_score': 20, 'game_date':'2024-09-14'},
        {'season': 2024, 'week': 2, 'home': 'DAL', 'away': 'WAS', 'home_score': 24, 'away_score': 17, 'game_date':'2024-09-15'},
    ])

# ------------------------------------------------------------------------------
# Entrenamiento ridge (margen del local con Elo + RPD)
# ------------------------------------------------------------------------------
def train_ridge(hist: pd.DataFrame):
    teams = pd.Index(hist['home']).append(pd.Index(hist['away'])).unique()
    elo = {t: 1500.0 for t in teams}

    # Snapshots Elo pre-juego y luego actualización
    records = []
    for (season, week), gdf in hist.sort_values(['season', 'week']).groupby(['season', 'week']):
        for _, g in gdf.iterrows():
            records.append({
                'season': season, 'week': week, 'home': g['home'], 'away': g['away'],
                'home_elo': elo.get(g['home'], 1500.0), 'away_elo': elo.get(g['away'], 1500.0)
            })
        for _, g in gdf.iterrows():
            elo = update_elo(elo, g['home'], g['away'], g['home_score'], g['away_score'])

    pre = pd.DataFrame(records)
    X = hist.merge(pre, on=['season', 'week', 'home', 'away'], how='left').copy()
    X['elo_diff'] = X['home_elo'] - X['away_elo']
    X['margin']   = X['home_score'] - X['away_score']

    rpd = rolling_point_diff(hist)  # team (cód), rpd
    X = X.merge(rpd.rename(columns={'team': 'home', 'rpd': 'home_rpd'}), on='home', how='left')
    X = X.merge(rpd.rename(columns={'team': 'away', 'rpd': 'away_rpd'}), on='away', how='left')
    X['rpd_diff'] = X['home_rpd'].fillna(0.0) - X['away_rpd'].fillna(0.0)

    feats = ['elo_diff', 'rpd_diff']  # intercepto lo maneja Ridge
    ridge = Ridge(alpha=1.0, fit_intercept=True).fit(X[feats], X['margin'])
    return ridge, feats, rpd, elo

# ------------------------------------------------------------------------------
# Odds (The Odds API)
# ------------------------------------------------------------------------------
def get_odds(odds_api_key: str) -> pd.DataFrame:
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {"regions": "us", "markets": "spreads", "oddsFormat": "american", "apiKey": odds_api_key}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    games = r.json()

    rows = []
    for g in games:
        home, away = g.get('home_team'), g.get('away_team')
        spread_home = None
        if 'bookmakers' in g and g['bookmakers']:
            prefer = ['DraftKings', 'Pinnacle', 'Caesars', 'FanDuel', 'BetMGM']
            sel = next((b for name in prefer for b in g['bookmakers'] if b.get('title') == name), g['bookmakers'][0])
            for mk in sel.get('markets', []):
                if mk.get('key') == 'spreads':
                    for o in mk.get('outcomes', []):
                        if o.get('name') == home:
                            try:
                                spread_home = float(o.get('point'))
                            except Exception:
                                spread_home = None
                            break
        if spread_home is None:
            continue
        rows.append({'home': home, 'away': away, 'market_spread_home': spread_home, 'commence_time': g.get('commence_time')})
    return pd.DataFrame(rows)

# ------------------------------------------------------------------------------
# SportsDataIO helpers (para lesiones)
# ------------------------------------------------------------------------------
def sdi_get_json(endpoint: str, key: str):
    url = "https://api.sportsdata.io/v3/nfl/" + endpoint.lstrip("/")
    headers = {"Ocp-Apim-Subscription-Key": key}
    try:
        r = requests.get(url, headers=headers, timeout=35)
        status = r.status_code
        ok = 200 <= status < 300
        try:
            data = r.json()
        except Exception:
            data = None
        if not ok:
            print(f"[sdi] {endpoint} -> HTTP {status} body_snip={str(r.text)[:180]}")
            return False, None
        if isinstance(data, dict) and data.get("Message"):
            print(f"[sdi] {endpoint} logical error: {data.get('Message')}")
            return False, None
        return True, data
    except Exception as e:
        print(f"[sdi] {endpoint} EXC: {e}")
        return False, None

def sdi_current_season_and_week(key: str):
    ok_s, cur_season = sdi_get_json("scores/json/CurrentSeason", key)
    ok_w, cur_week = sdi_get_json("scores/json/CurrentWeek", key)
    if ok_s and ok_w:
        try:
            return int(cur_season), int(cur_week)
        except Exception:
            return None, None
    return None, None

# ------------------------------------------------------------------------------
# Lesiones (dos fuentes) + diagnóstico SIEMPRE visible
# ------------------------------------------------------------------------------
def get_injuries_optional(sdi_key: str, team_codes: set) -> dict:
    """
    Retorna dict {code: {'qb': 'healthy|questionable|out'|'N/D', 'outs': N|'N/D'}}.
    Fuente por CFG['INJ_SOURCE']: 'scores_week' o 'projections'.
    """
    src_cfg = str(CFG.get("INJ_SOURCE", "projections")).lower()
    debug = bool(CFG.get("INJ_DEBUG", True))  # default True para diagnóstico

    if not sdi_key or not team_codes:
        return {t: {'qb': 'N/D', 'outs': 'N/D'} for t in team_codes}

    players = []
    used_source = src_cfg

    if src_cfg == "scores_week":
        season, week = sdi_current_season_and_week(sdi_key)
        if season and week:
            ok, data = sdi_get_json(f"scores/json/Injuries/{season}/{week}", sdi_key)
            if ok and isinstance(data, list):
                players = data
            else:
                used_source = "projections"
        else:
            used_source = "projections"

    if used_source == "projections" and not players:
        ok, data = sdi_get_json("projections/json/InjuredPlayers", sdi_key)
        if ok and isinstance(data, list):
            players = data

    out = {t: {'qb': 'N/D', 'outs': 'N/D'} for t in team_codes}
    teams_seen = set()

    for p in players or []:
        team = (p.get('Team') or '').upper()
        if team not in out:
            continue
        teams_seen.add(team)
        status = (p.get('InjuryStatus') or p.get('Status') or '').lower()
        pos    = (p.get('Position') or '').upper()

        if 'out' in status or 'doubtful' in status:
            out[team]['outs'] = 0 if out[team]['outs'] == 'N/D' else out[team]['outs']
            out[team]['outs'] += 1

        if pos == 'QB':
            if 'out' in status or 'doubtful' in status:
                out[team]['qb'] = 'out'
            elif 'questionable' in status or 'probable' in status:
                if out[team]['qb'] != 'out':
                    out[team]['qb'] = 'questionable'
            else:
                if out[team]['qb'] == 'N/D':
                    out[team]['qb'] = 'healthy'

    # SIEMPRE imprime un resumen para diagnosticar
    n_players = len(players or [])
    qb_flag = sum(1 for t,v in out.items() if v['qb'] in ('out','questionable'))
    outs_pos = sum(1 for t,v in out.items() if isinstance(v['outs'], (int,float)) and v['outs'] > 0)
    print(f"[inj] source={used_source} players={n_players} teams_with_data={len(teams_seen)} qb_flag={qb_flag} outs_pos={outs_pos}")

    if debug and n_players:
        sample = list(out.items())[:3]
        pretty = " | ".join([f"{k}:{v}" for k,v in sample])
        print(f"[inj] sample {pretty}")

    return out

def injury_points(qb_status: str, outs) -> float:
    q_out  = float(CFG.get('QB_OUT_PTS', 3.0))
    q_q    = float(CFG.get('QB_Q_PTS', 1.0))
    per_o  = float(CFG.get('STARTER_OUT_PTS', 0.3))
    cap    = float(CFG.get('INJURY_CAP_PTS', 5.0))
    qb = 0.0
    s = (qb_status or '').lower()
    if s == 'out': qb = q_out
    elif s == 'questionable': qb = q_q
    try: n_outs = float(outs)
    except Exception: n_outs = 0.0
    return float(min(qb + per_o * n_outs, cap))

# ------------------------------------------------------------------------------
# Features actuales (con CÓDIGOS)
# ------------------------------------------------------------------------------
def build_features_current(odds_df: pd.DataFrame, elo_now: dict, rpd_now: pd.DataFrame) -> pd.DataFrame:
    df = odds_df.copy()
    df['home_elo'] = df['home_code'].map(elo_now).fillna(1500.0)
    df['away_elo'] = df['away_code'].map(elo_now).fillna(1500.0)
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    r_home = rpd_now.rename(columns={'team': 'home_code', 'rpd': 'home_rpd'})
    r_away = rpd_now.rename(columns={'team': 'away_code', 'rpd': 'away_rpd'})
    df = df.merge(r_home, on='home_code', how='left').merge(r_away, on='away_code', how='left')
    df['rpd_diff'] = df['home_rpd'].fillna(0.0) - df['away_rpd'].fillna(0.0)
    return df

# ------------------------------------------------------------------------------
# Decisión de picks (probabilístico)
# ------------------------------------------------------------------------------
def decide_picks(df):
    alpha = float(CFG.get('PROB_ALPHA', 0.22))
    out = df.copy()
    out['edge_pts'] = out['model_spread_home_adj'] - out['market_spread_home']
    z = out['margin_pred_adj'] + out['market_spread_home']
    p_home_cover = 1.0 / (1.0 + np.exp(-alpha * z))
    out['p_home_cover'] = p_home_cover
    out['pick_side'] = np.where(out['p_home_cover'] >= 0.5, 'HOME', 'AWAY')
    out['p_cover_pick'] = np.where(out['pick_side'].eq('HOME'), out['p_home_cover'], 1 - out['p_home_cover'])
    out['confidence'] = out['p_cover_pick']
    return out

# ------------------------------------------------------------------------------
# Mensajes
# ------------------------------------------------------------------------------
def compose_messages(title: str, picks_df: pd.DataFrame, injuries: dict, chunk_limit: int = 3500) -> list:
    header = f"*{title}*"
    if picks_df.empty:
        return [header + "\n_No hay partidos disponibles._"]
    df = picks_df.sort_values('confidence', ascending=False).reset_index(drop=True)
    top_n = int(CFG.get("MAX_PICKS_DETAIL", 12))
    detailed = df.head(top_n)
    quick = df.iloc[top_n:]

    detail_blocks = []
    for _, r in detailed.iterrows():
        conf_pct = int(round(100 * float(r['confidence'])))
        p_cover_pct = int(round(100 * float(r['p_cover_pick'])))
        lines = []
        lines.append(f"{r['away']} @ {r['home']}")
        lines.append(f"Market (home): {r['market_spread_home']:+.1f} | Model: {r['model_spread_home_adj']:+.1f} | Edge: {r['edge_pts']:+.1f}")
        lines.append(f"Ajuste lesiones (home-away): {r['inj_diff_pts']:+.1f} pts")
        lines.append(f"Pick: *{r['pick_side']}* | Confianza: {conf_pct}% | Prob. cubrir: {p_cover_pct}%")
        h_code = r.get('home_code', None) or TEAM_MAP.get(r['home'])
        a_code = r.get('away_code', None) or TEAM_MAP.get(r['away'])
        if injuries and h_code in injuries and a_code in injuries:
            ih, ia = injuries[h_code], injuries[a_code]
            lines.append(f"Lesiones: {a_code} QB {ia['qb']}, outs {ia['outs']} vs {h_code} QB {ih['qb']}, outs {ih['outs']}")
        else:
            lines.append("Lesiones: N/D")
        lines.append("— Motivos: Elo y forma (RPD) + ajuste por lesiones.\n")
        detail_blocks.append("\n".join(lines))

    quick_blocks = []
    if not quick.empty:
        quick_blocks.append("— *Resto (menciones rápidas)* —")
        for _, r in quick.iterrows():
            conf_pct = int(round(100 * float(r['confidence'])))
            edge = f"{float(r['edge_pts']):+0.1f}"
            injd = f"{float(r['inj_diff_pts']):+0.1f}"
            quick_blocks.append(f"• {r['away']} @ {r['home']}: *{r['pick_side']}*, Conf {conf_pct}%, Edge {edge}, InjΔ {injd}")

    messages, current = [], header
    def try_add(block: str):
        nonlocal current, messages
        candidate = current + ("\n" if current else "") + block
        if len(candidate) > chunk_limit:
            messages.append(current)
            current = header + " (cont.)\n" + block
        else:
            current = candidate

    for b in detail_blocks: try_add(b)
    for b in quick_blocks: try_add(b)
    if current: messages.append(current)
    if len(messages) > 1:
        total = len(messages)
        messages = [f"{m}\n_Parte {i}/{total}_" for i, m in enumerate(messages, 1)]
    return messages

# ------------------------------------------------------------------------------
# Telegram
# ------------------------------------------------------------------------------
def send_telegram(token: str, chat_id: str, text: str, parse_mode: str = "Markdown"):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode, "disable_web_page_preview": True}
    r = requests.post(url, data=data, timeout=30)
    try: js = r.json()
    except Exception: js = {"raw": r.text}
    print(f"[telegram] status={r.status_code} ok={js.get('ok')} desc={js.get('description')}")
    r.raise_for_status()
    if not js.get('ok', False):
        raise RuntimeError(f"Telegram error: {js}")

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def main(batch: str):
    token   = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat_id = os.environ.get('TELEGRAM_CHAT_ID')
    odds_k  = os.environ.get('ODDS_API_KEY')
    sdio_k  = os.environ.get('SPORTSDATAIO_KEY', '')
    assert token and chat_id and odds_k, "Faltan TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID u ODDS_API_KEY"

    # 1) Entrena
    hist = load_hist()
    ridge, feats, rpd_now, elo_now = train_ridge(hist)
    print(f"[train] rows_hist={len(hist)} teams_elo={len(elo_now)} feats={feats}")

    # 2) Odds
    odds = get_odds(odds_k)
    if odds.empty:
        send_telegram(token, chat_id, "*NFL Picks*: no hay odds disponibles ahora mismo.")
        return

    # 3) Ventana temporal (America/Tijuana)
    tz = CFG.get("TIMEZONE", "America/Tijuana")
    odds['dt_utc'] = pd.to_datetime(odds['commence_time'], utc=True, errors='coerce')
    odds['dt_local'] = odds['dt_utc'].dt.tz_convert(tz)

    now_local = pd.Timestamp.now(tz=tz)
    if batch == 'tnf':
        days_to_thu = (3 - now_local.weekday()) % 7
        thu = (now_local + pd.Timedelta(days=days_to_thu)).normalize()
        start, end, title = thu, thu + pd.Timedelta(days=1), "TNF (ATS)"
        window_df = odds[(odds['dt_local'] >= start) & (odds['dt_local'] < end)].copy()
    else:
        days_to_fri = (4 - now_local.weekday()) % 7
        fri = (now_local + pd.Timedelta(days=days_to_fri)).normalize()
        mon = fri + pd.Timedelta(days=3)
        start, end, title = fri, mon + pd.Timedelta(days=1), "Weekend (ATS)"
        window_df = odds[(odds['dt_local'] >= start) & (odds['dt_local'] < end)].copy()

    window_df = window_df.sort_values('dt_local').drop_duplicates(subset=['home','away'], keep='first')
    window_df['home_code'] = window_df['home'].map(TEAM_MAP)
    window_df['away_code'] = window_df['away'].map(TEAM_MAP)
    before_len = len(window_df)
    window_df = window_df[window_df['home_code'].notna() & window_df['away_code'].notna()].copy()
    if before_len - len(window_df) > 0:
        print(f"[warn] descartados por mapeo desconocido: {before_len - len(window_df)}")

    if window_df.empty:
        send_telegram(token, chat_id, f"*{title}*: No hay partidos en la ventana {start.date()}–{(end - pd.Timedelta(seconds=1)).date()} ({tz}).")
        return

    # 4) Features base
    feats_df = build_features_current(window_df, elo_now, rpd_now)
    zero_elo = int((feats_df['elo_diff'] == 0).sum())
    zero_rpd = int((feats_df['rpd_diff'] == 0).sum())
    print(f"[diag] juegos={len(feats_df)} elo_diff=0:{zero_elo} rpd_diff=0:{zero_rpd}")

    margin_pred_base = ridge.predict(feats_df[['elo_diff','rpd_diff']])
    model_spread_home_base = -margin_pred_base

    # 5) Lesiones
    team_codes = set(pd.concat([feats_df['home_code'], feats_df['away_code']]).dropna().unique())
    injuries = get_injuries_optional(sdio_k, team_codes) if len(team_codes) > 0 else {}
    home_pts, away_pts = [], []
    for _, r in feats_df[['home_code', 'away_code']].iterrows():
        ih = injuries.get(r['home_code'], {'qb': 'N/D', 'outs': 'N/D'})
        ia = injuries.get(r['away_code'], {'qb': 'N/D', 'outs': 'N/D'})
        home_pts.append(injury_points(ih['qb'], ih['outs']))
        away_pts.append(injury_points(ia['qb'], ia['outs']))

    feats_df['home_inj_pts'] = home_pts
    feats_df['away_inj_pts'] = away_pts
    feats_df['inj_diff_pts'] = feats_df['home_inj_pts'] - feats_df['away_inj_pts']

    nonzero = int((feats_df['inj_diff_pts'] != 0).sum())
    print(f"[inj] nonzero_inj_diff_games={nonzero} of {len(feats_df)}")

    # 6) Ajuste final y picks
    margin_pred_adj = margin_pred_base - feats_df['inj_diff_pts']
    model_spread_home_adj = -margin_pred_adj

    out = feats_df.copy()
    out['home'] = window_df['home'].values
    out['away'] = window_df['away'].values
    out['market_spread_home']     = window_df['market_spread_home'].values
    out['margin_pred_adj']        = margin_pred_adj
    out['model_spread_home_adj']  = model_spread_home_adj

    out = decide_picks(out)

    # 7) Mensajes
    msgs = compose_messages(title, out, injuries, chunk_limit=3500)
    for m in msgs:
        send_telegram(token, chat_id, m, parse_mode="Markdown")

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch", choices=["tnf", "weekend"], required=True)
    args = p.parse_args()
    main(args.batch)
