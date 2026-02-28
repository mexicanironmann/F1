import fastf1 
import pandas as pd

fastf1.Cache.enable_cache("cache")

df = pd.read_csv("C:/Users/Owner/Desktop/F1/F1_Dataset.csv")

years = [2024, 2025]
rows = []


for year in years:
    for gp in range(1, 25):
        try:

            gp_data = {}
        ##FP2
            fp2 = fastf1.get_session(year, gp, "FP2")
            fp2.load(laps=True, weather=False)

            fp2_laps = fp2.laps

            fp2_stats = (
                fp2_laps.groupby("Driver")
                .agg(
                    fp2_TotalLaps = ("LapNumber", "count"),
                    fp2_FastestLap = ("LapTime", "min"),
                    fp2_AvgLap = ("LapTime", "mean")
                )
                .reset_index()
            )

            ##Quali
            quali = fastf1.get_session(year, gp, "Q")
            quali.load(laps = True, weather = False)

            quali_results = quali.results[["Abbreviation", "Position", "Q1", "Q2", "Q3"]].copy()
            quali_results.rename(columns= {
                "Abbreviation": "Driver",
                "Position": "Quali_Position",
                "Q1": "Q1_Time",
                "Q2": "Q2_Time",
                "Q3": "Q3_Time",
            }, inplace=True)

            ##Race
            race = fastf1.get_session(year, gp, "R")
            race.load(laps=True, weather=False)

            race_results = race.results[["Abbreviation", "Position", "Points", "Status"]].copy()
            race_results.rename(columns={
                "Abbreviation": "Driver",
                "Position": "Race_Position",
                "Points": "Race_Points",
                "Status": "Race_Status"
            }, inplace=True)

            race_laps = race.laps
            race_stats = (
                race_laps.groupby("Driver")
                .agg(
                    Race_TotalLaps = ("LapNumber", "count"),
                    Race_FastestLap = ("LapTime", "min"),
                    #Race_AvgLap = ("LapTime", "mean")
                
                ).reset_index()
            )
            
            combine = fp2_stats.merge(quali_results, on="Driver", how="outer")
            combine= combine.merge(race_results, on="Driver", how="outer")
            combine = combine.merge(race_stats, on="Driver", how="outer")

            combine["Year"] = year
            combine["Round"] = gp
            combine["GP_Name"] = fp2.event["EventName"]

            rows.append(combine)

            print(f"{year} {gp} {fp2.event['EventName']}")

        except Exception as e:
            print(f"FP2 failed: {e}")
            continue

if rows:
    new_data = pd.concat(rows, ignore_index=True)
    cols = ["Year", "Round", "GP_Name", "Driver", 
            "fp2_TotalLaps", "fp2_FastestLap", "fp2_AvgLap"
            "Quali_Position", "Q1_Time", "Q2_Time", "Q3_Time",
            "Race_Position", "Race_Points", "Race_Status",
            "Race_TotalLaps", "Race_FastestLap"]
    
    cols = [col for col in cols if col in new_data.columns]
    new_data = new_data[cols]

    df  =pd.concat([df, new_data], ignore_index=True)
    df.to_csv("C:/Users/Owner/Desktop/F1/F1_Dataset.csv", index=False)

print(df.head())
print(df.tail())