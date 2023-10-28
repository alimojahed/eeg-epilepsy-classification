import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt
import pickle
from scipy.signal import butter, lfilter
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import skew, norm, kurtosis, entropy
from sklearn.preprocessing import normalize
import random
import os
import pandas as pd

seed = 57


def pre_process_data_and_get():
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

    x = pickle.load(open('x.pkl', 'rb'))
    y = pickle.load(open('y.pkl', 'rb'))

    x_normal = np.concatenate((x[:300], x[400:]), axis=0)
    x_seizure = x[300:400]
    # print(x_normal.shape)
    # print(x_seizure.shape)
    sampling_freq = 173.6  # based on info from website

    b, a = butter(3, [0.5, 40], btype='bandpass', fs=sampling_freq)

    x_normal_filtered = np.array([lfilter(b, a, x_normal[ind, :]) for ind in range(x_normal.shape[0])])
    x_seizure_filtered = np.array([lfilter(b, a, x_seizure[ind, :]) for ind in range(x_seizure.shape[0])])
    # print(x_normal.shape)
    # print(x_seizure.shape)

    x_normal = x_normal_filtered
    x_seizure = x_seizure_filtered

    x = np.concatenate((x_normal, x_seizure))
    y = np.concatenate((np.zeros((400, 1)), np.ones((100, 1))))
    return x, y


def get_new_features(x):
    ptp = np.max(x) - np.min(x)
    twopp = np.argmax(x) - np.argmin(x)
    return np.array([
        np.mean(x),
        np.median(x),
        np.quantile(x, 0.25),
        np.quantile(x, 0.75),
        np.min(x),
        np.max(x),
        np.sum(x),
        np.sum(x ** 2),
        np.std(x),
        skew(x),
        kurtosis(x),
        entropy(np.unique(x, return_counts=True)[1], base=None),
        ptp,
        twopp,
        ptp / twopp
    ])


def get_pandas_df(new_x):
    return pd.DataFrame(new_x, columns=[
        'mean',
        'median',
        'q1',
        'q3',
        'min',
        'max',
        'sum',
        'sum of squares',
        'std',
        'skew',
        'kurtosis',
        'entropy',
        'ptp',
        'twopp',
        'ptp/twopp'
    ])


def get_cross_validation_result(model, x, y, k=5):
    return cross_validate(estimator=model,
                          X=x,
                          y=y,
                          cv=k,
                          scoring=['accuracy', 'precision', 'recall'],
                          return_train_score=False)


def get_data_with_new_features(x, do_normalize=True):
    new_x = [get_new_features(arr) for arr in x]
    new_x = np.array(new_x)

    if do_normalize:
        new_x = normalize(new_x)

    return new_x


def main():
    x, y = pre_process_data_and_get()
    new_x = get_data_with_new_features(x, True)
    # x_train, x_test, y_train, y_test = train_test_split(new_x, y, random_state=seed, test_size=0.2)

    # print(x_test.shape)

    clf = SVC()
    # clf.fit(x_train, y_train)
    #
    # y_pred = clf.predict(x_test)

    svm_k_fold_result = get_cross_validation_result(SVC(kernel='rbf'), new_x, y)
    # svm_k_fold_result = get_cross_validation_result(SVC(kernel='linear'), new_x, y)
    # svm_k_fold_result = get_cross_validation_result(SVC(kernel='poly'), new_x, y)
    # svm_k_fold_result = get_cross_validation_result(SVC(kernel='sigmoid'), new_x, y)

    print("svm accuracy is : " + str(svm_k_fold_result['test_accuracy'].mean()) +
          " precision is : " + str(svm_k_fold_result['test_precision'].mean()) +
          "recall is: " + str(svm_k_fold_result['test_recall'].mean()))

    knn_k_fold_result = get_cross_validation_result(KNeighborsClassifier(n_neighbors=5), new_x, y)
    # knn_k_fold_result = get_cross_validation_result(KNeighborsClassifier(n_neighbors=30), new_x, y)
    # knn_k_fold_result = get_cross_validation_result(KNeighborsClassifier(n_neighbors=100), new_x, y)
    # knn_k_fold_result = get_cross_validation_result(KNeighborsClassifier(n_neighbors=500), new_x, y)

    print("knn accuracy is : " + str(knn_k_fold_result['test_accuracy'].mean()) +
          " precision is : " + str(knn_k_fold_result['test_precision'].mean()) +
          "recall is: " + str(knn_k_fold_result['test_recall'].mean()))

    # rf_k_fold_result = get_cross_validation_result(RandomForestClassifier(random_state=0, max_depth=3), new_x, y)
    # rf_k_fold_result = get_cross_validation_result(RandomForestClassifier(random_state=0, max_depth=2), new_x, y)
    rf_k_fold_result = get_cross_validation_result(RandomForestClassifier(random_state=0, max_depth=5), new_x, y)

    print("random forest accuracy is : " + str(rf_k_fold_result['test_accuracy'].mean()) +
          " precision is : " + str(rf_k_fold_result['test_precision'].mean()) +
          "recall is: " + str(rf_k_fold_result['test_recall'].mean()))

    best_clf = RandomForestClassifier(random_state=0, max_depth=5)
    x_train, x_test, y_train, y_test = train_test_split(new_x, y, random_state=seed, test_size=0.2)
    best_clf.fit(x_train, y_train)
    y_pred = best_clf.predict(x_test)
    cfm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cfm).plot()

    RocCurveDisplay.from_predictions(y_test, y_pred).plot()
    plt.show()


if __name__ == "__main__":
    main()
