import tensorflow as tf
import keras
from keras import layers, models, losses, optimizers


def create_ffn(input_size, class_names):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=input_size))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(units=1 if len(class_names) == 2 else len(class_names),
                           activation='sigmoid' if len(class_names) == 2 else 'softmax'))
    optimizer = optimizers.Adam()
    model.compile(optimizer, loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model


def train_ffn(X_train, y_train, X_test, y_test, epochs, model: models.Sequential):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model, history


def train_ffn_ds(train_ds, test_ds, epochs, model: models.Sequential):
    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    return model, history
