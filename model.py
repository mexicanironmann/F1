import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib as plt

##Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("C:/Users/Owner/Desktop/F1/F1_Dataset.csv")


##Create Target 
df["Top3"] = (df["Race_Position"] <= 3).astype("Int64")


##Rolling features
df = df.sort_values(["Driver", "Year", "GP_Name"]).reset_index(drop=True)
rolling_cols = {
    "Race_Position": [3, 5, 10],
    "Race_Points": [3, 5, 10]
}

for col, windows in rolling_cols.items():
    if col not in df.columns:
        continue
    for i in windows:
        df[f"{col}_roll{i}"] = (
            df.groupby("Driver")[col].transform(lambda x: x.shift(1).rolling(i, min_periods = 1).mean())
        )

for i in [3, 5, 10]:
    df[f"Top3_rate_roll{i}"] = (
        df.groupby("Driver")["Top3"].transform(lambda x: x.shift(1).rolling(i, min_periods = 1).mean())
    )

#print(df.head())
print(df.columns)


categorical_data = ["GP_Name", "Driver", "Race_Status"]

label_encoders = {}
for col in categorical_data:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

#Overfitting issues
drop_cols = ["Race_Position", "Race_Points", "Race_Status", "Race_TotalLaps", "Race_FastestLap", "Top3", "Year"]

train = df[df["Year"] <= 2022] 
val = df[df["Year"] == 2023]
test = df[df["Year"] >= 2024]

X_train = train.drop(drop_cols, axis=1)
y_train = train["Top3"]

X_val = val.drop(drop_cols, axis=1)
y_val = val["Top3"]

X_test = test.drop(drop_cols, axis=1)
y_test = test["Top3"]


params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'learning_rate': 0.1,
    'n_estimators': 500,
    'max_depth': 5,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
    'random_state': 42

    
}


model = xgb.XGBClassifier(**params)
model.fit(
    X_train, y_train, eval_set = [(X_val, y_val)],
    verbose = True
)

Top3 = model.predict_proba(X_test)[:, 1]

#print(Top3)


results = test.copy()
results["Top3_Prob"] = Top3


results["Driver"] = label_encoders["Driver"].inverse_transform(results["Driver"])
results["GP_Name"] = label_encoders["GP_Name"].inverse_transform(results["GP_Name"])

final_output = []

for (year, gp), group in results.groupby(["Year", "GP_Name"]):
    top3_drivers = (
        group.sort_values("Top3_Prob", ascending=False).head(3)["Driver"].tolist()
    )

    print(f"{gp} {year}")
    for i, driver in enumerate(top3_drivers, 1):
        print(f"     {i}. {driver}")
        print()


##Evaluation 
accuracy = accuracy_score(y_test, model.predict(X_test))
precision = precision_score(y_test, model.predict(X_test))
recall = recall_score(y_test, model.predict(X_test))
f1 = f1_score(y_test, model.predict(X_test))

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"f1: {f1}")
