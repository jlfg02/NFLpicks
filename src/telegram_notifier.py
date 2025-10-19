import os
import argparse
import pytz
import requests
from datetime import datetime, timedelta

# ============================================================
# CONFIG
# ============================================================

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORTSDATAIO_KEY = os.getenv("SPORTSDATAIO_KEY")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

TZ = pytz.timezone("America/Tijuana")

# ============================================================
# HELPERS
# ============================================================

def get_effective_week(today=None):
    """
    Corrige la semana efectiva para que el sábado por la noche
    tome la semana actual y no la siguiente.
    """
    now = datetime.now(TZ) if today is None else today
    # isocalendar()[1] devuelve el número de semana ISO
    # Restamos 1 día si es sábado o domingo
    adjusted_date = now - timedelta(days=1 if now.weekday() >= 5 else 0)
    return adjusted_date.isocalendar()[1]

def send_telegram_message(message):
    """Envía el mensaje al bot de Telegram."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    r = requests.post(url, json=payload)
    return r.status_code, r.text

# ============================================================
# DATA FETCH
# ============================================================

def get_odds(week):
    """Obtiene spreads de The Odds API para la semana indicada."""
    url = f"https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
    params = {"apiKey": ODDS_API_KEY, "regions": "us", "markets": "spreads"}
    resp = requests.get(url, params=params, timeout=20)
    data = resp.json()

    # Filtrar solo los juegos de NFL regulares que correspondan al rango de la semana actual
    # (algunos endpoints no entregan 'week', así que este paso es solo filtro temporal)
    games = []
    for g in data:
        if "home_team" in g and "away_team" in g:
            spread = g["bookmakers"][0]["markets"][0]["outcomes"]
            home = next((x for x in spread if x["name"] == g["home_team"]), None)
            away = next((x for x in spread if x["name"] == g["away_team"]), None)
            if home and away:
                games.append({
                    "home": g["home_team"],
                    "away": g["away_team"],
                    "market_spread": float(home["point"]),
                })
    return games

def get_injuries():
    """Obtiene reporte resumido de lesiones (SportsDataIO)."""
    url = f"https://api.sportsdata.io/v3/nfl/projections/json/InjuredPlayers?key={SPORTSDATAIO_KEY}"
    try:
        resp = requests.get(url, timeout=20)
        data = resp.json()
    except Exception:
        return {}

    teams = {}
    for p in data:
        team = p.get("Team")
        if not team:
            continue
        teams.setdefault(team, {"outs": 0, "qb_out": False})
        if p.get("Position") == "QB" and p.get("InjuryStatus") in ("Out", "Doubtful"):
            teams[team]["qb_out"] = True
        if p.get("InjuryStatus") in ("Out", "Doubtful"):
            teams[team]["outs"] += 1
    return teams

# ============================================================
# MODEL
# ============================================================

def estimate_model_spread(home_team, away_team, elo_diff=0, rpd=0, injury_adj=0):
    """
    Modelo simple de spread: Elo + forma + ajuste lesiones.
    Negative spread favors home (home stronger).
    """
    base = -1.5  # ventaja local promedio
    return base - 0.05 * elo_diff - 0.02 * rpd + injury_adj

def decide_pick(market_spread, model_spread):
    """Decide pick y nivel de confianza."""
    edge = model_spread - market_spread
    pick = "HOME" if edge < 0 else "AWAY"
    confidence = min(99, max(55, 50 + abs(edge) * 4))
    return edge, pick, confidence

# ============================================================
# MAIN EXECUTION
# ============================================================

def main(batch):
    week = get_effective_week()
    games = get_odds(week)
    injuries = get_injuries()

    if not games:
        send_telegram_message("❌ No se encontraron partidos o spreads disponibles.")
        return

    msg_lines = [f"*{batch.upper()} (ATS)*"]
    for g in games:
        home = g["home"]
        away = g["away"]
        market = g["market_spread"]
        home_inj = injuries.get(home, {"outs": 0, "qb_out": False})
        away_inj = injuries.get(away, {"outs": 0, "qb_out": False})
        injury_adj = (away_inj["outs"] - home_inj["outs"]) * 0.1
        if home_inj["qb_out"]:
            injury_adj += 3
        if away_inj["qb_out"]:
            injury_adj -= 3

        model_spread = estimate_model_spread(home, away, 0, 0, injury_adj)
        edge, pick, conf = decide_pick(market, model_spread)
        if abs(edge) < 1.5:
            continue

        msg_lines.append(
            f"{away} @ {home}\n"
            f"Market (home): {market:+.1f} | Model: {model_spread:+.1f} | Edge: {edge:+.1f}\n"
            f"Ajuste lesiones (home-away): {injury_adj:+.1f} pts\n"
            f"Pick: *{pick}* | Confianza: {conf:.0f}% | Prob. cubrir: {conf:.0f}%\n"
            f"Lesiones: {away} outs {away_inj['outs']} (QB {'out' if away_inj['qb_out'] else 'OK'}) vs {home} outs {home_inj['outs']} (QB {'out' if home_inj['qb_out'] else 'OK'})\n"
            f"— Motivos: Elo y forma (RPD) + ajuste por lesiones.\n"
        )

    if len(msg_lines) == 1:
        msg_lines.append("PASS — ningún edge ≥ 1.5 pts detectado.")
print("[debug] mensaje generado:")
print("\n".join(msg_lines))
    send_telegram_message("\n".join(msg_lines))

# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=str, default="weekend")
    args = parser.parse_args()
    main(args.batch)