import nfl_data_py as nfl
import pandas as pd

keep_cols = [
    #identifiers
    'play_id', 'game_id', 'season', 'week',

    #teams/context
    'home_team', 'away_team', 'posteam', 'defteam',

    #game situation
    'quarter_seconds_remaining', 'half_seconds_remaining', 'game_seconds_remaining', 'qtr',
    'total_home_score', 'total_away_score', 'score_differential',
    'season_type',

    #kicker & outcome
    'kicker_player_name', 'kicker_player_id',
    'field_goal_result', 'kick_distance',

    #environment
    'stadium', 'game_stadium', 'roof', 'surface', 'temp', 'wind', 'weather',

    #optional metadata
    'game_date', 'location',

    #timeouts / icing
    'home_timeouts_remaining', 'away_timeouts_remaining',
    'posteam_timeouts_remaining', 'defteam_timeouts_remaining',
    'timeout', 'timeout_team',

    #vegas lines / game context
    'spread_line', 'total_line', 'vegas_wp', 'vegas_home_wp'
]

pbp = nfl.import_pbp_data(list(range(2013, 2025)))

fg = pbp[pbp['play_type'] == 'field_goal']

fg_clean = fg[keep_cols]

fg_clean.to_csv("nfl_field_goals_clean_2013_2024.csv", index=False)

print(f"Saved {len(fg_clean)} rows and {len(keep_cols)} columns to parquet.")