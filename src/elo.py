import numpy as np

def expected_score(rating_a, rating_b, home_advantage=0.0):
    return 1.0 / (1.0 + 10 ** (-(rating_a + home_advantage - rating_b)/400.0))

def update_elo(elo, home, away, home_points, away_points, k=20.0, cap_margin=14, home_advantage=55):
    margin = max(min(home_points - away_points, cap_margin), -cap_margin)
    result = 1.0 if home_points > away_points else 0.0 if home_points < away_points else 0.5
    exp = expected_score(elo.get(home,1500.0), elo.get(away,1500.0), home_advantage=home_advantage)
    scale = 1.0 + abs(margin)/cap_margin
    delta = k * scale * (result - exp)
    elo[home] = elo.get(home,1500.0) + delta
    elo[away] = elo.get(away,1500.0) - delta
    return elo
