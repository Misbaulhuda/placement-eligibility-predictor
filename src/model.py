from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(df):

    X = df.drop("Placement_Status", axis=1)
    y = df["Placement_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)

    return model, accuracy, X.columns
