import os, requests, pandas as pd, numpy as np, yaml
from sklearn.linear_model import Ridge, LogisticRegression
from pathlib import Path
from elo import update_elo
from features import rolling_point_diff

CFG = yaml.safe_load(open(Path(__file__).resolve().parents[1]/'config.yaml'))

def load_hist():
    p = Path(__file__).resolve().parents[1] / 'data' / 'historical_games.csv'
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame([
        {'season':2024,'week':1,'home':'KC','away':'DET','home_score':20,'away_score':21,'closing_spread_home':-6.5,'game_date':'2024-09-07'},
        {'season':2024,'week':1,'home':'NYJ','away':'BUF','home_score':22,'away_score':16,'closing_spread_home':2.5,'game_date':'2024-09-08'},
        {'season':2024,'week':2,'home':'KC','away':'CIN','home_score':27,'away_score':20,'closing_spread_home':-3.0,'game_date':'2024-09-14'},
        {'season':2024,'week':2,'home':'DAL','away':'WAS','home_score':24,'away_score':17,'closing_spread_home':-4.0,'game_date':'2024-09-15'},
    ])

def train_models(hist):
    teams = pd.Index(hist['home']).append(pd.Index(hist['away'])).unique()
    elo = {t:1500.0 for t in teams}
    records = []
    for (season, week), gdf in hist.sort_values(['season','week']).groupby(['season','week']):
        for _, g in gdf.iterrows():
            records.append({'season':season,'week':week,'home':g['home'],'away':g['away'],
                            'home_elo':elo.get(g['home'],1500.0),'away_elo':elo.get(g['away'],1500.0)})
        for _, g in gdf.iterrows():
            elo = update_elo(elo, g['home'], g['away'], g['home_score'], g['away_score'])
    pre = pd.DataFrame(records)
    X = hist.merge(pre, on=['season','week','home','away'], how='left').copy()
    X['elo_diff'] = X['home_elo'] - X['away_elo']
    X['margin'] = X['home_score'] - X['away_score']
    X['home_cover'] = (X['margin'] + X['closing_spread_home'] >= 0).astype(int)
    rpd = rolling_point_diff(hist)
    X = X.merge(rpd.rename(columns={'team':'home','rpd':'home_rpd'}), on='home', how='left')
    X = X.merge(rpd.rename(columns={'team':'away','rpd':'away_rpd'}), on='away', how='left')
    X['rpd_diff'] = X['home_rpd'].fillna(0.0) - X['away_rpd'].fillna(0.0)
    X['home_flag'] = 1
    feats = ['elo_diff','rpd_diff','home_flag']
    logit = LogisticRegression(max_iter=200).fit(X[feats], X['home_cover'])
    ridge = Ridge(alpha=1.0).fit(X[feats], X['margin'])
    return logit, ridge, feats, rpd, elo

def get_odds(api_key):
    url = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/"
    params = {"regions":"us","markets":"spreads","oddsFormat":"american","apiKey":api_key}
    r = requests.get(url, params=params, timeout=30); r.raise_for_status()
    games = r.json()
    rows = []
    for g in games:
        home, away = g.get('home_team'), g.get('away_team')
        spread_home = None
        if 'bookmakers' in g and g['bookmakers']:
            prefer = ['DraftKings','Pinnacle','Caesars','FanDuel','BetMGM']
            sel = next((b for name in prefer for b in g['bookmakers'] if b.get('title')==name), g['bookmakers'][0])
            for mk in sel.get('markets', []):
                if mk.get('key') == 'spreads':
                    for o in mk.get('outcomes', []):
                        if o.get('name') == home:
                            spread_home = float(o.get('point')); break
        if spread_home is None: 
            continue
        rows.append({'home':home,'away':away,'market_spread_home':spread_home,'commence_time':g.get('commence_time')})
    return pd.DataFrame(rows)

def build_features_current(odds_df, elo_now, rpd_now):
    df = odds_df.copy()
    df['home_elo'] = df['home'].map(elo_now).fillna(1500.0)
    df['away_elo'] = df['away'].map(elo_now).fillna(1500.0)
    df['elo_diff'] = df['home_elo'] - df['away_elo']
    r_home = rpd_now.rename(columns={'team':'home','rpd':'home_rpd'})
    r_away = rpd_now.rename(columns={'team':'away','rpd':'away_rpd'})
    df = df.merge(r_home, on='home', how='left').merge(r_away, on='away', how='left')
    df['rpd_diff'] = df['home_rpd'].fillna(0.0) - df['away_rpd'].fillna(0.0)
    df['home_flag'] = 1
    return df

def decide_picks(df, edge_pts):
    out = df.copy()
    out['edge_pts'] = out['model_spread_home'] - out['market_spread_home']
    picks, confs = [], []
    for _, r in out.iterrows():
        side = None
        if abs(r['edge_pts']) >= edge_pts:
            side = 'HOME' if r['edge_pts'] < 0 else 'AWAY'
        conf = min(1.0, abs(r['edge_pts'])/(edge_pts+1e-9))
        picks.append(side); confs.append(conf)
    out['pick_side'] = picks; out['confidence'] = confs
    return out

def compose_message(title, picks_df):
    lines = [f"*{title}*"]
    if picks_df.empty or picks_df['pick_side'].isna().all():
        lines.append(f"_Sin edge ≥ {CFG['EDGE_PTS']} pts. PASS._")
        return "\n".join(lines)
    top = picks_df.sort_values('confidence', ascending=False).head(3)
    for _, r in top.iterrows():
        lines.append(f"{r['away']} @ {r['home']}")
        lines.append(f"Market (home): {r['market_spread_home']:+.1f} | Model: {r['model_spread_home']:+.1f} | Edge: {r['edge_pts']:+.1f}")
        lines.append(f"Pick: *{r['pick_side']}* | Confianza: {r['confidence']:.2f}")
        lines.append("— Motivos: Elo y forma (RPD); lesiones si disponibles.\n")
    return "\n".join(lines).strip()

def send_telegram(msg, token, chat_id, parse_mode="Markdown"):
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": msg, "parse_mode": parse_mode, "disable_web_page_preview": True}
    r = requests.post(url, data=data, timeout=30); r.raise_for_status()

def main(batch:str):
    token = os.environ.get('TELEGRAM_BOT_TOKEN')
    chat = os.environ.get('TELEGRAM_CHAT_ID')
    odds_key = os.environ.get('ODDS_API_KEY')
    assert token and chat and odds_key, "Faltan TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID o ODDS_API_KEY"
    hist = load_hist()
    logit, ridge, feats, rpd_now, elo_now = train_models(hist)
    odds = get_odds(odds_key)
    if odds.empty:
        send_telegram("*NFL Picks*: no hay odds disponibles ahora mismo.", token, chat); return
    odds['dt'] = pd.to_datetime(odds['commence_time'], utc=True, errors='coerce')
    odds['weekday'] = odds['dt'].dt.weekday  # Thu=3, Fri=4, Sun=6, Mon=0
    if batch == 'tnf':
        df = odds[odds['weekday'] == 3].copy(); title = "TNF (ATS)"
    else:
        df = odds[odds['weekday'].isin([4,5,6,0])].copy(); title = "Weekend (ATS)"
    if df.empty:
        send_telegram(f"*{title}*: No hay partidos en ventana esperada.", token, chat); return
    feats_df = build_features_current(df, elo_now, rpd_now)
    feats = ['elo_diff','rpd_diff','home_flag']
    feats_df['model_spread_home'] = -ridge.predict(feats_df[feats])
    out = decide_picks(feats_df, CFG['EDGE_PTS'])
    msg = compose_message(title, out)
    send_telegram(msg, token, chat)

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch", choices=["tnf","weekend"], required=True)
    args = p.parse_args()
    main(args.batch)
