import keras.models


class Hybrid:
    def __init__(self, steps: []):
        self.steps = steps

    def fit(self, X_train, y_train):
        for s in self.steps:
            if isinstance(s, keras.models.Sequential):
                X_train = s.predict(X_train)
            else:
                pass
        self.steps[len(self.steps) - 1].fit(X_train, y_train.ravel())

    def predict(self, X):
        for s in self.steps:
            X = s.predict(X)
        return X
