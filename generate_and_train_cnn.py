import tensorflow as tf
import keras
from keras import layers, models, losses, optimizers


def create_cnn(input_size, class_names):
    model = models.Sequential()
    model.add(layers.Conv2D(16, kernel_size=(3, 3), padding='same',
                            activation='relu', input_shape=input_size))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same',
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same',
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='valid',
                            activation='relu'))
    model.add(layers.MaxPooling2D())
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(units=1 if len(class_names) == 2 else len(class_names),
                           activation='sigmoid' if len(class_names) == 2 else 'softmax'))

    optimizer = optimizers.Adam()
    model.compile(optimizer, loss=losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    return model

def train_cnn(X_train, y_train, X_test, y_test, epochs, model: models.Sequential):
    history = model.fit(X_train, y_train, epochs=epochs, validation_data=(X_test, y_test))
    return model, history


def train_cnn_ds(train_ds, test_ds, epochs, model: models.Sequential):
    history = model.fit(train_ds, epochs=epochs, validation_data=test_ds)
    return model, history
