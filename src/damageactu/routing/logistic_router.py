from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def fit_logistic_router(X, y):
    model = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=2000, random_state=42
        )),
    ])
    return model.fit(X, y)
