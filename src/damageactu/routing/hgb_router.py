from sklearn.ensemble import HistGradientBoostingClassifier


def fit_hgb_router(X, y):
    model = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    return model.fit(X, y)
