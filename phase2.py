import matplotlib.pyplot as plt
import numpy as np

from experiments import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score
from sklearn.cluster import KMeans


def main():
    x, y = pre_process_data_and_get()
    new_x = get_data_with_new_features(x)
    x = new_x
    df = get_pandas_df(new_x)
    corr = df.corr()
    plt.figure(figsize=(30, 15))
    plt.matshow(corr)
    # https://stackoverflow.com/questions/3529666/matplotlib-matshow-labels
    plt.xticks(range(df.shape[1]), df.columns, rotation=90)
    plt.yticks(range(df.shape[1]), df.columns)

    plt.colorbar()
    plt.show()

    features_f1 = [get_f1_score_for_each_feature(x, y, index, df.columns[index]) for index in range(15)]
    features_f1.sort(key=lambda x: -x[0])
    print(features_f1)
    desire_number_of_selected_feature = 8
    desire_accuracy = 0.9
    accuracy = 0
    number_of_selected_features = 0
    selected_features = [features_f1[0][1]]

    while accuracy < desire_accuracy or number_of_selected_features <= desire_number_of_selected_feature:
        correlation_with_current = []
        for index in range(15):
            if index in selected_features:
                continue

            correlation_with_current.append(get_correlation(corr, index, selected_features))

        correlation_with_current.sort(key=lambda x: x[0])
        # print(correlation_with_current)
        for _, index in correlation_with_current:
            temp_selected_feature = list(selected_features)
            temp_selected_feature.append(index)
            x_train, x_test, y_train, y_test = train_test_split(x[:, temp_selected_feature], y, test_size=0.2,
                                                                random_state=seed, shuffle=True)
            clf = RandomForestClassifier(random_state=0).fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            accuracy = accuracy_score(y_test, y_pred)

            if accuracy > desire_accuracy:
                number_of_selected_features += 1
                selected_features.append(index)
                print(f'select {df.columns[index]} with accuracy {accuracy}')
                break

    selected_cols = [df.columns[i] for i in selected_features]
    print(f'selected {selected_cols} with final accuracy of {accuracy}')

    x_train, x_test, y_train, y_test = train_test_split(x[:, selected_features], y, test_size=0.2,
                                                        random_state=seed, shuffle=True)

    clustering, labels = cluster(x_train)
    clustering_acc = classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test)
    print(f'accuracy with clustering {clustering_acc}')

    best_x = x[:, selected_features]

    # a vs c
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((best_x[0:100], best_x[200:300])),
                                                        np.concatenate((np.zeros(100), np.ones(100))), test_size=0.2,
                                                        random_state=seed, shuffle=True)
    clustering, labels = cluster(x_train)
    clustering_acc = classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test)
    print(f"a vs c class {clustering_acc} ")

    # a vs e
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((best_x[0:100], best_x[400:500])),
                                                        np.concatenate((np.zeros(100), np.ones(100))), test_size=0.2,
                                                        random_state=seed, shuffle=True)
    clustering, labels = cluster(x_train)
    clustering_acc = classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test)
    print(f"a vs e class {clustering_acc} ")

    # b vs e
    x_train, x_test, y_train, y_test = train_test_split(np.concatenate((best_x[100:200], best_x[400:500])),
                                                        np.concatenate((np.zeros(100), np.ones(100))), test_size=0.2,
                                                        random_state=seed, shuffle=True)
    clustering, labels = cluster(x_train)
    clustering_acc = classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test)
    print(f"b vs e class {clustering_acc} ")

    # ab cd e
    x_train, x_test, y_train, y_test = train_test_split(
        np.concatenate((best_x[0:200], best_x[200:400], best_x[400:500])),
        np.concatenate((np.zeros(200), np.ones(200), np.full(100, 2))), test_size=0.2,
        random_state=seed, shuffle=True)
    clustering, labels = cluster(x_train)
    clustering_acc = classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test)
    print(f"ab vs cd vs e class {clustering_acc} ")


def cluster(x_train, cluster=5):
    kmeans = KMeans(n_clusters=cluster, random_state=0, n_init="auto").fit(x_train)
    return kmeans, kmeans.labels_


def classify_with_clustered_data(clustering, labels, x_train, x_test, y_train, y_test, cluster=5):
    classifiers = []

    for i in range(cluster):
        x = x_train[labels == i]
        y = y_train[labels == i]
        clf = RandomForestClassifier(random_state=0, max_depth=5).fit(x, y)
        classifiers.append(clf)

    predictions = []
    for data in x_test:
        label = clustering.predict(data.reshape(1, -1))
        pred = classifiers[label[0]].predict(data.reshape(1, -1))
        predictions.append(pred)

    return accuracy_score(y_test, np.array(predictions))


def get_correlation(correlation_matrix, feature_index, current_features):
    # not_correlated = 0
    # correlation_threshold = 0.5
    #
    # for index in current_features:
    #     if abs(correlation_matrix.iloc[feature_index, index]) < correlation_threshold:
    #         not_correlated += 1
    #
    # return not_correlated, feature_index

    sum = 0
    for index in current_features:
        sum += abs(correlation_matrix.iloc[feature_index, index])

    return sum, feature_index


def get_f1_score_for_each_feature(x, y, feature_index, title):
    x_train, x_test, y_train, y_test = train_test_split(x[:, feature_index].reshape(-1, 1),
                                                        y,
                                                        test_size=0.2,
                                                        random_state=seed)
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    f1 = f1_score(y_test, y_pred)

    return f1, feature_index, title


if __name__ == "__main__":
    main()
