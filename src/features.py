import pandas as pd

def rolling_point_diff(hist_df, window=4, cap=14):
    rows = []
    for _, r in hist_df.iterrows():
        rows.append({'team': r['home'], 'pd': max(min(r['home_score']-r['away_score'], cap), -cap), 'season': r['season'], 'week': r['week']})
        rows.append({'team': r['away'], 'pd': max(min(r['away_score']-r['home_score'], cap), -cap), 'season': r['season'], 'week': r['week']})
    t = pd.DataFrame(rows).sort_values(['team','season','week'])
    t['rpd'] = t.groupby('team')['pd'].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    return t.groupby('team', as_index=False).last()[['team','rpd']]
