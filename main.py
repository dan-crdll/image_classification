import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import keras
import generate_and_train_ffn
import generate_and_train_cnn
import show_plots
import pickle
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from hybrid_model import Hybrid


max_n = 2000

# section: DATASET LOADING AND PREPROCESS
DB = keras.datasets.cifar10
(X_train, y_train), (X_test, y_test) = DB.load_data()

X_train, X_test = X_train / 255.0, X_test / 255.0
X_shape = X_train.shape
input_shape = (X_shape[1], X_shape[2], X_shape[3])

# | 0 | airplane | | 1 | automobile | | 2 | bird | | 3 | cat |
# | 4 | deer | | 5 | dog | | 6 | frog | | 7 | horse |
# | 8 | ship | | 9 | truck |
class_names = ['airplane', 'automobile', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# section: FFN MODEL AND EVALUATION
if os.path.exists('./models/ffn_model.keras'):
    ffn_model = keras.models.load_model('./models/ffn_model.keras')
else:
    ffn_model = generate_and_train_ffn.create_ffn(input_shape, class_names)
    ffn_model, ffn_history = generate_and_train_ffn.train_ffn(X_train, y_train,
                                                              X_test, y_test, 10, ffn_model)
    show_plots.show_accuracy_plots(ffn_history)

print(ffn_model.summary())

if not os.path.exists('./models/ffn_model.keras'):
    ffn_model.save('./models/ffn_model.keras')

show_plots.show_confusion_matrix(ffn_model, X_test, y_test)

# section: CNN MODEL AND EVALUATION
if os.path.exists('./models/cnn_model.keras'):
    cnn_model = keras.models.load_model('./models/cnn_model.keras')
else:
    cnn_model = generate_and_train_cnn.create_cnn(input_shape, class_names)
    cnn_model, cnn_history = generate_and_train_cnn.train_cnn(X_train, y_train,
                                                              X_test, y_test, 10, cnn_model)
    show_plots.show_accuracy_plots(cnn_history)

print(cnn_model.summary())

if not os.path.exists('./models/cnn_model.keras'):
    cnn_model.save('./models/cnn_model.keras')

show_plots.show_confusion_matrix(cnn_model, X_test, y_test)

# section: SVM MODEL AND EVALUATION
if os.path.exists('./models/svm_model.sav'):
    svm_model = pickle.load(open('./models/svm_model.sav', 'rb'))
else:
    svm_model = SVC()
    svm_model.fit(X_train[:max_n].reshape((
        max_n,
        X_shape[1] * X_shape[2] * X_shape[3]
    )), y_train.ravel()[:max_n])

if not os.path.exists('./models/svm_model.sav'):
    pickle.dump(svm_model, open('./models/svm_model.sav', 'wb'))

show_plots.show_confusion_matrix(svm_model, X_test.reshape((
        X_test.shape[0],
        X_shape[1] * X_shape[2] * X_shape[3]
    )), y_test.ravel())

# section: HYBRID SVM AND CNN MODEL AND EVALUATION
if os.path.exists('./models/hybrid_model.sav'):
    hybrid_model = pickle.load(open('./models/hybrid_model.sav', 'rb'))
else:
    feature_extractor_step = keras.models.load_model('./models/cnn_model.keras')
    for i in range(0, 5):
        feature_extractor_step.pop()
    svm_model_step = SVC()

    hybrid_model = Hybrid([feature_extractor_step, svm_model_step])
    hybrid_model.fit(X_train, y_train)

if not os.path.exists('./models/hybrid_model.sav'):
    pickle.dump(hybrid_model, open('./models/hybrid_model.sav', 'wb'))

show_plots.show_confusion_matrix(hybrid_model, X_test, y_test)
