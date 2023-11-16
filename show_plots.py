import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.svm
from sklearn.metrics import confusion_matrix
import numpy as np

import hybrid_model


def show_accuracy_plots(history):
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    plt.plot(train_accuracy, label='Train Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(loc='lower right')

    plt.show()


def show_confusion_matrix(model, X_test, y_test):
    y_pred = []
    if not (isinstance(model, sklearn.svm.SVC) or isinstance(model, hybrid_model.Hybrid)):
        pred = model.predict(X_test)
        for p in pred:
            if len(p > 1):
                y_pred.append(np.argmax(p))
            else:
                y_pred.append(1 if pred > 0.5 else 0)
        y_pred = np.array(y_pred)
    else:
        y_pred = model.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm, cmap='Purples', annot=True, fmt='.2f')
    plt.ylabel('Predicted')
    plt.xlabel('true')
    plt.show()
