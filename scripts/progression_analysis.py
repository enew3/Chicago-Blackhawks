#Evan New 
#The goal of this project is to analyze when central vs wide ball progression leads to a more favorable outcome

#import necessary modules
import os
import contextlib
import pandas as pd
import numpy as np
from models import load_expected_goals_model, calculate_xg_features

#split pitch into zones long ways with 30-40-30 split
X_MAX = 120
Y_MAX = 80
WIDE_LEFT_MAX = Y_MAX * 0.30   #~24 yards
CENTRAL_MAX   = Y_MAX * 0.70   #~56 yards

#create function to load and combine data from simulate csv files
def load_data(data):
    #loads all csv files in directory
    files = [i for i in os.listdir(data) if i.endswith(".csv")]
    #only load columns we need
    columns = ["GAME_ID", "TIMESTEP", "PLAYER_ID", "TEAM_ID", "LOCATION_X", "LOCATION_Y", "VELOCITY_X", "VELOCITY_Y", "ON_BALL", "TEAM_0_SCORE", "TEAM_1_SCORE"]
    #join all files into one. three games per file. sort by game and then time with resetting index to be cleaner
    allgames = pd.concat([pd.read_csv(os.path.join(data, i), usecols=columns, dtype="float32")for i in files]).sort_values(by=["GAME_ID", "TIMESTEP"]).reset_index(drop=True)
    return allgames

#function to convert from the coordinates given to real world system as the pitch is 120x80 yds
def scale_coordinates(allgames):
    #given is [-1, 1] for x and [-0.42, 0.42] for y. convert to [0, 120] and [0, 80]. also scale the velocities to be in yds/s
    allgames = allgames.copy()
    allgames["LOCATION_X"] = ((allgames["LOCATION_X"] + 1) / 2) * X_MAX
    allgames["LOCATION_Y"] = ((allgames["LOCATION_Y"] + 0.42) / 0.84) * Y_MAX
    allgames["VELOCITY_X"] = allgames["VELOCITY_X"] * X_MAX
    allgames["VELOCITY_Y"] = allgames["VELOCITY_Y"] * Y_MAX
    return allgames

#function that labels what zone the ball is in based on y coordnate from the 30-40-30 split
def what_zone(y):
    if y < WIDE_LEFT_MAX:
        return "wide"
    elif y > CENTRAL_MAX:
        return "wide"
    else:
        return "central"

#function that labels what zone the ball is in based on x coordinate from tradition 33-33-33 split in soccer
def field_third(x, team_id):
    #always from the attacking teams perspective
    if team_id == 0:
        attack_x = x
    else:
        attack_x = X_MAX - x
    #three thirds split equally across the 120 yards
    if attack_x < X_MAX / 3:
        return "defensive_third"
    elif attack_x < 2 * X_MAX / 3:
        return "middle_third"
    else:
        return "final_third"

#possession is counted as a progression if it moved the ball forward by at least 10 yards
MIN_FORWARD_YARDS   = 10
#13.93 yards given by finding the median of the defensive width (standard deviation of defending team y-coordinates)
DEF_WIDTH_THRESHOLD = 13.93

#ufunction to determine possesions
def possessions(df: pd.DataFrame, ball: pd.DataFrame) -> pd.DataFrame:
    """
    Identifies possession sequences -- runs of consecutive timesteps where the same team has the ball.

    How it works:
      1. Find all timesteps where a player has ON_BALL=True and grab their TEAM_ID
      2. Join that onto the ball position data (one ball row per timestep)
      3. Detect when possession switches from one team to the other
      4. Assign each run of same-team possession a unique POSSESSION_ID

    Returns one row per timestep with the possessing team and a POSSESSION_ID label.
    Timesteps where no player has the ball (loose ball) get TEAM_ID = NaN.
    """
    #find time where a player had the ON_BALL column as true and get their team id
    on_ball = (df[df["ON_BALL"] == True][["GAME_ID", "TIMESTEP", "TEAM_ID"]].drop_duplicates(subset=["GAME_ID", "TIMESTEP"]))
    #find position of the ball in the moments found with on_ball
    ball_position = ball[["GAME_ID", "TIMESTEP", "LOCATION_X", "LOCATION_Y", "ZONE"]].copy()
    #combine the two to get what team has the ball at each timestamp
    possesion_id = ball_position.merge(on_ball, on=["GAME_ID", "TIMESTEP"], how="left")
    #find instances where the team with the ball changes (possesion swap) or when the game changes (new game)
    possesion_change = possesion_id["TEAM_ID"] != possesion_id["TEAM_ID"].shift(1)
    new_game = possesion_id["GAME_ID"] != possesion_id["GAME_ID"].shift(1)
    #get a unique id for every possession by summing every time the possesion changes or a new game starts
    possesion_id["POSSESSION_CHANGE"] = possesion_change | new_game
    possesion_id["POSSESSION_ID"] = possesion_id["POSSESSION_CHANGE"].cumsum()
    return possesion_id

#function that determines if each possesion counts as a progression based on the forward distance moved (at least 10 yards)
def find_progressions(possession_df: pd.DataFrame, players: pd.DataFrame)->  pd.DataFrame:
    """
    Collapses each possession sequence from many timestep rows into one summary row.

    For each possession sequence we calculate:
      - Which team had it and for how long
      - How far the ball moved forward (in the attacking direction)
      - Where on the pitch it started (field third)
      - Which zone the ball was in most of the time (dominant zone)
      - Which zone the ball entered and exited through
      - DEF_WIDTH: average Y spread (std dev) of the defending team

    A sequence is classified as a "progression" if it moved the ball forward
    by at least MIN_FORWARD_YARDS. This filters out backwards passes and
    possessions that go nowhere.

    Returns two DataFrames:
      possessions  - all possession sequences (including non-progressions)
      progressions - only the forward-moving ones (DISTANCE >= MIN_FORWARD_YARDS)
    """
    rows = []
    #sort players by game and time to make life easier
    players_by_game = {}
    for i, j in players.groupby("GAME_ID"):
        players_by_game[i] = j.sort_values("TIMESTEP").reset_index(drop=True)
    #group possesions by game and possesion id to be able to loop through each possesion
    for (game_id, poss_id), group in possession_df.groupby(["GAME_ID", "POSSESSION_ID"]):
        team_id = group["TEAM_ID"].iloc[0]
        #if there was no team in posession (NaN) then skip possesion entirely
        if pd.isna(team_id):
            continue
        #find starting and ending x coordinate of the ball in a possesion
        start_x = group["LOCATION_X"].iloc[0]
        end_x   = group["LOCATION_X"].iloc[-1]
        #team 0 attacks toward larger x  and Team 1 toward smaller x
        if team_id == 0:
            distance = end_x - start_x
        else:
            distance = start_x - end_x
        #determine the field third that the attack starts, ends, and spends the most time in
        third        = field_third(start_x, int(team_id))
        dominant_zone = group["ZONE"].value_counts().idxmax()
        entry_zone   = group["ZONE"].iloc[0]
        exit_zone    = group["ZONE"].iloc[-1]
        #find defensive width (spread of defending team) during possesion by look at defending players y-coordinates and finding the standard deviatoin
        def_width = 0.0
        getplayers = players_by_game.get(game_id)
        if getplayers is not None and not getplayers.empty:
            tmin = group["TIMESTEP"].iloc[0]
            tmax = group["TIMESTEP"].iloc[-1]
            delta_t = getplayers["TIMESTEP"].values
            min = np.searchsorted(delta_t, tmin, side="left")
            max = np.searchsorted(delta_t, tmax, side="right")
            range = getplayers.iloc[min:max]
            defender = range[range["TEAM_ID"] != team_id]
            if not defender.empty:
                width_per = defender.groupby("TIMESTEP")["LOCATION_Y"].std().dropna()
                if not width_per.empty:
                    def_width = float(width_per.mean())

        rows.append({
            "GAME_ID": game_id,
            "POSSESSION_ID": poss_id,
            "TEAM_ID": int(team_id),
            "START_X": start_x,
            "START_Y": group["LOCATION_Y"].iloc[0],
            "END_X": end_x,
            "DISTANCE": distance,
            "THIRD": third,
            "DOMINANT_ZONE": dominant_zone,
            "ENTRY_ZONE": entry_zone,
            "EXIT_ZONE": exit_zone,
            "START_TIME": group["TIMESTEP"].iloc[0],
            "END_TIME": group["TIMESTEP"].iloc[-1],
            "DEF_WIDTH": round(def_width, 3),
        })
    #locate the possessions and the progressions
    possessions  = pd.DataFrame(rows)
    progressions = possessions[possessions["DISTANCE"] >= MIN_FORWARD_YARDS].copy()
    return possessions, progressions

#determine number of timestamps after a progression to see if a shot/goal occured. Choose about 5 seconds arbitraririly based on soccer knowledge
SHOT_WINDOW = 50
#ball needs to be within 5 yards of goal to count as a shot and have velocity towards the goal (coded later)
IS_SHOT = 5

#function to label the outcome of each progressoion
def label_outcomes(progressions, possessions, ball, xg_model):
    results = []
    #group who has ball and who is in possession by game to make life easier
    ball_by_game = {i: j for i, j in ball.groupby("GAME_ID")}
    poss_by_game = {i: j for i, j in possessions.groupby("GAME_ID")}
    #loop through every progession to determine what event occured at the end of it
    for j, prog in progressions.iterrows():
        game_id = prog["GAME_ID"]
        team_id = prog["TEAM_ID"]
        end_time= prog["END_TIME"]
        possession_id = prog["POSSESSION_ID"]
        game_ball = ball_by_game.get(game_id, pd.DataFrame())
        #find window after end of progression to look for event based on the 5 second amount chosen earlier
        ball_distance = game_ball[(game_ball["TIMESTEP"] > end_time) & (game_ball["TIMESTEP"] <= (end_time + SHOT_WINDOW))]
        #detect shot based on the distance of ball to goal and velocity towards goal. take into account both teams
        if team_id == 0:
            shots= (ball_distance["LOCATION_X"] >= X_MAX - IS_SHOT) & (ball_distance["VELOCITY_X"] > 0)
        else:
            shots = (ball_distance["LOCATION_X"] <= IS_SHOT) & (ball_distance["VELOCITY_X"] < 0)
        #count all rows that are true
        shot = shots.any()
        #if there is a shot, claculate the xG using the given, pre-trained model
        xg = 0.0
        if shot:
            shot_place = ball_distance[shots]
            if team_id == 0:
                shot_row = shot_place.loc[shot_place["LOCATION_X"].idxmax()].copy()
            else:
                shot_row = shot_place.loc[shot_place["LOCATION_X"].idxmin()].copy()
            #make data frame needed for the xG model
            xG_df = pd.DataFrame([{"TEAM_ID": team_id, "LOCATION_X": shot_row["LOCATION_X"], "LOCATION_Y": shot_row["LOCATION_Y"], "IS_GOAL": False}])
            #predict xG using the model
            with contextlib.redirect_stdout(None):
                X, _ = calculate_xg_features(xG_df)
            xg = xg_model.predict_proba(X)[0, 1]
        #check if a goal was scored following the progression
        score_before = game_ball[game_ball["TIMESTEP"] == end_time]
        score_after  = game_ball[(game_ball["TIMESTEP"] > end_time) & (game_ball["TIMESTEP"] <= (end_time + SHOT_WINDOW))]
        goal = False
        if len(score_before) > 0 and len(score_after) > 0:
            if team_id == 0:
                goal = score_after["TEAM_0_SCORE"].max() > score_before["TEAM_0_SCORE"].iloc[0]
            else:
                goal = score_after["TEAM_1_SCORE"].max() > score_before["TEAM_1_SCORE"].iloc[0]
        #determine if there was a turnover by checking if the opposing team gained posession
        game_poss = poss_by_game.get(game_id, pd.DataFrame())
        next_poss = game_poss[game_poss["POSSESSION_ID"] > possession_id].sort_values("POSSESSION_ID")
        turnover = False
        if len(next_poss) > 0:
            next_team = next_poss.iloc[0]["TEAM_ID"]
            if not shot and not pd.isna(next_team) and int(next_team) != team_id:
                turnover = True
        #add the results of this function to the data frame
        results.append({"GAME_ID": game_id, "POSSESSION_ID": possession_id, "SHOT": shot, "XG": xg, "GOAL": goal, "TURNOVER": turnover,})
    #conbime the two results to have it all
    outcomes = pd.DataFrame(results)
    return progressions.merge(outcomes, on=["GAME_ID", "POSSESSION_ID"])

#function to include the context of the games
def features(progressions, ball):
    #get the score at the start of each progression by merging game id and time
    scores = ball[["GAME_ID", "TIMESTEP", "TEAM_0_SCORE", "TEAM_1_SCORE"]].copy()
    scores = scores.rename(columns={"TIMESTEP": "START_TIME"})
    progressions = progressions.merge(scores, on=["GAME_ID", "START_TIME"], how="left")
    #function to determine if the team was winning, losing, or tied at the beginning of the progression
    def score_state(row):
        if row["TEAM_ID"] == 0:
            diff = row["TEAM_0_SCORE"] - row["TEAM_1_SCORE"]
        else:
            diff = row["TEAM_1_SCORE"] - row["TEAM_0_SCORE"]
        if diff > 0:
            return "winning"
        elif diff < 0: 
            return "losing"
        else:          
            return "tied"
    #apply function and add score state to dataframe
    progressions["SCORE_STATE"] = progressions.apply(score_state, axis=1)
    return progressions

#function to determine defensive width as compact or stretched based on the median that was found earlier
def categorize_def_width(progressions):
    progressions["DEF_WIDTH_CATEGORY"] = progressions["DEF_WIDTH"].apply(lambda w: "compact" if w < DEF_WIDTH_THRESHOLD else "stretched")
    return progressions


#function to calculate created metric of progression value (PV)
#PV = mean XG - (turnover rate x opponent's expected XG after receiving the ball)
def compute_PV(progressions):
    #create mapping to find opposite third. makes calculating easier
    opposite_third = {"defensive_third": "final_third", "middle_third": "middle_third", "final_third": "defensive_third",}
    #find opponent xG after receiving the ball based on third and wide/central
    zonal_xG = progressions.groupby(["THIRD", "DOMINANT_ZONE"])["XG"].mean()
    rows = []
    #loop through each of the 18 scenarios to calcualte an average PV for each
    for (third, score_state, def_width_cat, zone), group in progressions.groupby(["THIRD", "SCORE_STATE", "DEF_WIDTH_CATEGORY", "DOMINANT_ZONE"]):
        mean_xg = group["XG"].mean()
        turnover_rate = group["TURNOVER"].mean()
        opp_third = opposite_third[third]
        turnover_cost = zonal_xG.get((opp_third, zone), 0.0)
        pv = mean_xg - (turnover_rate * turnover_cost)
        #add results to data frame
        rows.append({
            "THIRD": third,
            "SCORE_STATE": score_state,
            "DEF_WIDTH_CATEGORY": def_width_cat,
            "ZONE": zone,
            "MEAN_XG": mean_xg,
            "TURNOVER_RATE": turnover_rate,
            "TURNOVER_COST": turnover_cost,
            "PV": pv,
            "COUNT": len(group),
        })
    pv_df = pd.DataFrame(rows)
    return pv_df

#checkpoint to save the data frame which makes running much faster when looking at the visualizations and lookup function
CHECKPOINT = "events/progressions_checkpoint.csv"

#function to produce deliverable where user inputs scenario and gets PV output for wide and central
def lookup_pv(pv_df):
    thirds = ["defensive_third", "middle_third", "final_third"]
    third_labels = ["Defensive Third", "Middle Third", "Final Third"]
    scores = ["losing", "tied", "winning"]
    score_labels = ["Losing", "Tied", "Winning"]
    defs = ["compact", "stretched"]
    def_labels = ["Compact", "Stretched"]
    print("Progression Value Calculator")
    while True:
        print("Which third?")
        for i, l in enumerate(third_labels, 1):
            print(f"  {i}. {l}")
        third = thirds[int(input("> ").strip()) - 1]
        print("Score state?")
        for i, l in enumerate(score_labels, 1):
            print(f"  {i}. {l}")
        score_idx = int(input("> ").strip()) - 1
        score = scores[score_idx]
        print("Defensive formation?")
        for i, l in enumerate(def_labels, 1):
            print(f"  {i}. {l}")
        def_idx = int(input("> ").strip()) - 1
        def_cat = defs[def_idx]
        subset = pv_df[(pv_df["THIRD"] == third) & (pv_df["SCORE_STATE"] == score) & (pv_df["DEF_WIDTH_CATEGORY"] == def_cat)]
        cent_pv = subset[subset["ZONE"] == "central"]["PV"].values[0]
        wide_pv = subset[subset["ZONE"] == "wide"]["PV"].values[0]
        preferred = "Central" if cent_pv > wide_pv else "Wide"
        advantage = abs(cent_pv - wide_pv)
        third_label = third_labels[thirds.index(third)]
        print("Results:")
        print(f"Scenario: {third_label} | {score_labels[score_idx]} | {def_labels[def_idx]} Defense")
        print(f"Central PV: {cent_pv:+.4f}")
        print(f"Wide PV: {wide_pv:+.4f}")
        print(f"{preferred} progression is preferred({advantage:.4f} advantage)")
        if input("\nLook up another scenario? (y/n): ").strip().lower() != "y":
            break

#execute the code to run the analysis and produce the visualizations
if __name__ == "__main__":
    xg_model = load_expected_goals_model("tmp/models/xg/xg_model.pkl")
    progressions = pd.read_csv(CHECKPOINT)
    progressions = categorize_def_width(progressions)
    pv_df = compute_PV(progressions)
    lookup_pv(pv_df)