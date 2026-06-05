import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)

    df["CGPA"] = df["CGPA"].fillna(df["CGPA"].median())

    mode_skill = df["Technical_Skills"].mode()[0]
    df["Technical_Skills"] = df["Technical_Skills"].fillna(mode_skill)

    return df


def engineer_features(df):

    skill_dummies = df["Technical_Skills"].str.get_dummies(sep=",")

    skill_dummies.columns = skill_dummies.columns.str.strip()

    df = pd.concat([df, skill_dummies], axis=1)

    df.drop(["Student_ID", "Technical_Skills"], axis=1, inplace=True)

    scaler = StandardScaler()

    df["CGPA"] = scaler.fit_transform(df[["CGPA"]])

    return df, scaler, skill_dummies.columns.tolist()
