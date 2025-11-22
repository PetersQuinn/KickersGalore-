import pandas as pd
#sourced from the start.py script to only carry relevant columns forward for quicker script execution
pbp = pd.read_parquet("nfl_field_goals_clean_2013_2024.parquet")
#Forgot to drop this column in the original script
pbp = pbp.drop(columns=['timeout_team'])
#noticed that all indoor stadiums had NaN for temp and wind, so filling those in with typical values
pbp['temp'] = pbp['temp'].fillna(68)
pbp['wind'] = pbp['wind'].fillna(0)
#There were about 180 weather entries that were NaN, so filling those in with "Unknown"
#the weather column is super messy with a lot of different formats, so we will likely have to revisit this later.
pbp['weather'] = pbp['weather'].fillna("Unknown")

""" We need to now try to turn this data into numerical data that can be used in a model
there are a ton of high cardinality categorical variables that we will need to deal with
 we will start with the easy ones"""

#we will turn the season type into a binary variable and drop the categorical version
pbp['season_type_binary'] = (pbp['season_type'] == 'POST').astype(int)
pbp = pbp.drop(columns=['season_type'])

"""we will turn the field goal result into a binary variable and drop the categorical version
first, we must make a choice on what to do with blocked kicks
often, blocked kicks are not on the kicker, but rather a failure of the blocking scheme
we will drop blocked kicks for now, but we may want to revisit this later"""

pbp = pbp[pbp['field_goal_result'] != 'blocked']
pbp['field_goal_result_binary'] = (pbp['field_goal_result'] == 'made').astype(int)
pbp = pbp.drop(columns=['field_goal_result'])

# there are two high cardinality categorical variables in stadium and game_stadium
# we can easily track if the roof is open or not, but altitude is an important factor in kicking not currently captured. 
#we encode open roof as 0 and closed/dome as 1
pbp['roof_binary'] = (pbp['roof'].isin(['closed', 'dome'])).astype(int)
pbp = pbp.drop(columns=['roof'])


#now let's look at the surface column
pbp['surface'] = pbp['surface'].str.strip().str.lower()
#map categories simply to turf or grass
pbp['surface'] = pbp['surface'].replace({
    'fieldturf': 'turf',
    'sportturf': 'turf',
    'matrixturf': 'turf',
    'astroturf': 'turf',
    'a_turf': 'turf',
    'astroplay': 'turf',
    'grass': 'grass',
    '': 'unknown'
})
#we have to handle the unknown surfaces. they are all from 2022 on, and many are from international games
# there are about 176 which means deleting 176 rows from the most recent years could be a more significant deletion.
#unfortunately, we will have to do some manual mapping based on stadium and year

unknown_surfaces = pbp[pbp['surface'] == 'unknown']
stadium_surface_map = {
    ('Acrisure Stadium', 2023): 'turf',
    ('AT&T Stadium', 2023): 'turf',
    ('Bank of America Stadium', 2023): 'grass',
    ('Caesars Superdome', 2023): 'turf',
    ('Cleveland Browns Stadium', 2023): 'grass',
    ('Commanders Field', 2023): 'grass',
    ('Empower Field at Mile High', 2023): 'grass',
    ('EverBank Stadium', 2023): 'grass',
    ('Ford Field', 2023): 'turf',
    ('GEHA Field at Arrowhead Stadium', 2023): 'grass',
    ('Gillette Stadium', 2023): 'grass',
    ('Highmark Stadium', 2023): 'grass',
    ("Levi's® Stadium", 2022): 'grass',
    ('Lincoln Financial Field', 2023): 'grass',
    ('Lucas Oil Stadium', 2023): 'turf',
    ('Lumen Field', 2023): 'grass',
    ('M&T Bank Stadium', 2023): 'grass',
    ('Mercedes-Benz Stadium', 2023): 'turf',
    ('MetLife Stadium', 2023): 'turf',
    ('NRG Stadium', 2023): 'turf',
    ('Nissan Stadium', 2023): 'grass',
    ('Paycor Stadium', 2023): 'grass',
    ('Raymond James Stadium', 2023): 'grass',
    ('SoFi Stadium', 2023): 'turf',
    ('Soldier Field', 2023): 'grass',
    ('State Farm Stadium', 2023): 'grass',
    ('U.S. Bank Stadium', 2023): 'turf',
    ('Estadio Azteca (Mexico City)', 2022): 'grass',
    ('Wembley Stadium', 2022): 'grass',
    ('Wembley Stadium', 2023): 'grass',
    ('Tottenham Hotspur Stadium', 2022): 'grass',
    ('Tottenham Hotspur Stadium', 2023): 'grass',
    ('Allianz Arena', 2022): 'grass',
    ('Allianz Arena', 2024): 'grass',
    ('Arena Corinthians', 2024): 'grass',
}
#apply the mapping to fill in the unknown surfaces
mask = pbp['surface'] == 'unknown'
pbp.loc[mask, 'surface'] = pbp.loc[mask].apply(
    lambda row: stadium_surface_map.get((row['stadium'], row['season']), 'unknown'),
    axis=1
)
# now we mark the grass as 1 and turf as 0, and drop the surface column
pbp['surface_binary'] = (pbp['surface'] == 'grass').astype(int)
pbp = pbp.drop(columns=['surface'])

#speaking of stadiums, we will have to do a similar map for the altitude of the stadiums and if it was international or not
stadium_map = {
    "Empower Field at Mile High": {"altitude": 5280, "international": 0},
    "Sports Authority Field at Mile High": {"altitude": 5280, "international": 0},
    "Estadio Azteca (Mexico City)": {"altitude": 7200, "international": 1},
    "Azteca Stadium": {"altitude": 7200, "international": 1},
    "Arena Corinthians": {"altitude": 2400, "international": 1},
    "Allianz Arena": {"altitude": 1715, "international": 1},  
    "Frankfurt Stadium": {"altitude": 385, "international": 1}, 
    "Deutsche Bank Park": {"altitude": 385, "international": 1},
    "Wembley Stadium": {"altitude": 90, "international": 1},
    "Tottenham Hotspur Stadium": {"altitude": 75, "international": 1},
    "Tottenham Stadium": {"altitude": 75, "international": 1},
    "Twickenham Stadium": {"altitude": 70, "international": 1},
    "Rogers Centre": {"altitude": 250, "international": 1},  
    "Soldier Field": {"altitude": 594, "international": 0},
    "Lambeau Field": {"altitude": 640, "international": 0},
    "Highmark Stadium": {"altitude": 600, "international": 0},
    "Arrowhead Stadium": {"altitude": 1020, "international": 0},
    "GEHA Field at Arrowhead Stadium": {"altitude": 1020, "international": 0},
    "FirstEnergy Stadium": {"altitude": 653, "international": 0},
    "Cleveland Browns Stadium": {"altitude": 653, "international": 0},
    "Paycor Stadium": {"altitude": 482, "international": 0},
    "Paul Brown Stadium": {"altitude": 482, "international": 0},
    "Heinz Field": {"altitude": 730, "international": 0},
    "Acrisure Stadium": {"altitude": 730, "international": 0},
    "Ford Field": {"altitude": 605, "international": 0},
    "Lucas Oil Stadium": {"altitude": 720, "international": 0},
    "M&T Bank Stadium": {"altitude": 20, "international": 0},
    "FedExField": {"altitude": 280, "international": 0},
    "Commanders Field": {"altitude": 280, "international": 0},
    "Bank of America Stadium": {"altitude": 748, "international": 0},
    "MetLife Stadium": {"altitude": 3, "international": 0},
    "Caesars Superdome": {"altitude": 3, "international": 0},
    "Mercedes-Benz Superdome": {"altitude": 3, "international": 0},
    "Mercedes-Benz Stadium": {"altitude": 1050, "international": 0},
    "State Farm Stadium": {"altitude": 1070, "international": 0},
    "University of Phoenix Stadium": {"altitude": 1070, "international": 0},
    "NRG Stadium": {"altitude": 50, "international": 0},
    "Reliant Stadium": {"altitude": 50, "international": 0},
    "AT&T Stadium": {"altitude": 430, "international": 0},
    "Raymond James Stadium": {"altitude": 0, "international": 0},
    "Gillette Stadium": {"altitude": 289, "international": 0},
    "Lincoln Financial Field": {"altitude": 10, "international": 0},
    "Hard Rock Stadium": {"altitude": 7, "international": 0},
    "Sun Life Stadium": {"altitude": 7, "international": 0},
    "Nissan Stadium": {"altitude": 430, "international": 0},
    "LP Field": {"altitude": 430, "international": 0},
    "EverBank Stadium": {"altitude": 16, "international": 0},
    "EverBank Field": {"altitude": 16, "international": 0},
    "TIAA Bank Stadium": {"altitude": 16, "international": 0},
    "Levi's® Stadium": {"altitude": 15, "international": 0},
    "Levi's Stadium": {"altitude": 15, "international": 0},
    "Los Angeles Memorial Coliseum": {"altitude": 233, "international": 0},
    "SoFi Stadium": {"altitude": 100, "international": 0},
    "Allegiant Stadium": {"altitude": 2001, "international": 0},
    "CenturyLink Field": {"altitude": 20, "international": 0},
    "Lumen Field": {"altitude": 20, "international": 0},
    "Georgia Dome": {"altitude": 1000, "international": 0},
    "Edward Jones Dome": {"altitude": 465, "international": 0},
    "Mall of America Field": {"altitude": 840, "international": 0},
    "TCF Bank Stadium": {"altitude": 900, "international": 0},
    "U.S. Bank Stadium": {"altitude": 840, "international": 0},
    "New Era Field": {"altitude": 600, "international": 0},
    "Ring Central Coliseum": {"altitude": 43, "international": 0},
    "O.co Coliseum": {"altitude": 43, "international": 0},
    "Oakland-Alameda County Coliseum": {"altitude": 43, "international": 0},
    "Oakland-Alameda County Stadium": {"altitude": 43, "international": 0},
    "StubHub Center": {"altitude": 220, "international": 0},
    "ROKiT Field - Dignity Health Sports Park": {"altitude": 220, "international": 0},
    "Northwest Stadium": {"altitude": 0, "international": 0},  
    "Huntington Bank Field": {"altitude": 0, "international": 0}, 
}
pbp['altitude'] = pbp['stadium'].map(lambda x: stadium_map.get(x, {}).get('altitude', 0))
pbp['international'] = pbp['stadium'].map(lambda x: stadium_map.get(x, {}).get('international', 0))
pbp = pbp.drop(columns=['stadium', 'game_stadium'])

#now we clean up the vegas_wp. there are two columns that are redundant and can be shortened to just one column.
pbp['vegas_wp_effective'] = pbp.apply(
    lambda row: row['vegas_home_wp'] if row['posteam'] == row['home_team'] else row['vegas_wp'],
    axis=1
)
#drop the old columns and total over/under, which is very unlikely to be useful
pbp = pbp.drop(columns=['vegas_wp', 'vegas_home_wp', "total_line"])

#we can also reduce the columns for score to simply differential and total. 
pbp['total_points_scored'] = pbp['total_home_score'] + pbp['total_away_score']
pbp = pbp.drop(columns=['total_home_score', 'total_away_score'])


"""we need to do some cleaning on the weather column
it's a mess with 3000+ unique values, so we will have to extract some useful features from it
we already have temperature and wind, so we will focus on precipitation"""

pbp['weather'] = pbp['weather'].str.lower().fillna("")
#extract rain/snow indicators
pbp['is_rain'] = pbp['weather'].str.contains("rain").astype(int)
pbp['is_snow'] = pbp['weather'].str.contains("snow|flurries|sleet|hail|blizzard").astype(int)
pbp = pbp.drop(columns=['weather', 'location'])

#we can also drop a couple more columns that are unlikely to be useful
pbp = pbp.drop(columns=['week', 'home_team', 'away_team', 'posteam', 'defteam',])

"""we know that who the kicker is will have a big impact on the outcome, but there are too many kickers to one-hot encode
we will replace the kicker with their career field goal percentage up to that point in time
kicker_history.csv was manually compiled from pro-football-reference.com"""

pre2013_lookup = pd.read_csv("kicker_history.csv")
pbp = pbp.merge(pre2013_lookup, how="left", on="kicker_player_name")
pbp['pre2013_fga'] = pbp['pre2013_fga'].fillna(0)
pbp['pre2013_fgm'] = pbp['pre2013_fgm'].fillna(0)
pbp['pre2013_fg_pct'] = pbp['pre2013_fgm'] / pbp['pre2013_fga'].replace(0, 1)
pbp = pbp.sort_values(by=['kicker_player_id', 'game_date'])
pbp['career_attempts'] = (
    pbp.groupby('kicker_player_id').cumcount() + pbp['pre2013_fga']
)
pbp['career_makes'] = (
    pbp.groupby('kicker_player_id')['field_goal_result_binary'].cumsum()
    - pbp['field_goal_result_binary'] + pbp['pre2013_fgm']
)

#to avoid division by zero, we replace 0 attempts with 1 temporarily
pbp['career_fg_pct'] = pbp['career_makes'] / pbp['career_attempts'].replace(0, 1)
pbp = pbp.drop(columns=['game_date', 'timeout', 'home_timeouts_remaining', 'away_timeouts_remaining', 'kicker_player_id', 'pre2013_fga', 'pre2013_fgm', 'pre2013_fg_pct'])

#now we need to rid of noisy columns that won't be helpful moving forward
pbp = pbp.drop(columns=['play_id', 'game_id', 'quarter_seconds_remaining', 'half_seconds_remaining', 'posteam_timeouts_remaining', 'defteam_timeouts_remaining'])
#we can encode 4th quarter and buzzer beater variables as something that may be impactful for prediction in the future
pbp['is_4th_qtr'] = (pbp['qtr'] == 4).astype(int)
pbp['buzzer_beater_binary'] = (pbp['game_seconds_remaining'] <= 5).astype(int)
pbp = pbp.drop(columns=['game_seconds_remaining', 'qtr'])

# just realized that career_makes is redudnant with career_attempts and career_fg_pct, so we drop it
pbp = pbp.drop(columns=['career_makes'])
# also total_points_scored is redundant with score_differential, so we drop it
pbp = pbp.drop(columns=['total_points_scored'])

"""as well, with professionals, you would expect that spread_line would not be predicitve of field goal success, so we drop it
the argument to keeping it would only be that if you expect to win by a lot and are not, kickers may feel more pressure to cover for their team
however, again, with professionals we would not expect them to consider the spread
vegas_wp_effective already captures the expected competitiveness of the game, so we drop spread_line"""

pbp = pbp.drop(columns=['spread_line'])

#as well, it will be really tough to generalize if we include international, so we will drop that
pbp = pbp.drop(columns=['international'])

#now we are done
#there are a ton of lines here that don't need to run everytime, so we will save the cleaned data to a new file
#this will be the starting point for the next script
pbp.to_csv("field_goals_model_ready.csv", index=False)
pbp.to_parquet("field_goals_model_ready.parquet", index=False)

print("Saved as field_goals_model_ready.csv and field_goals_model_ready.parquet")
pbp.info()
pbp.head()