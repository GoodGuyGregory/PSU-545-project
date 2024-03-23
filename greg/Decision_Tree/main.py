import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


def getBloodFeatures():
    df = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')
    df = df.drop('Disease', axis=1)
    featureNames = df.columns.tolist()
    return featureNames

def loadTrainingData():
    trainingData = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')
    # shuffle the training data
    trainingData.sample(frac=1)
    # replace the labels as numeric values
    trainingLabels = trainingData.iloc[:, -1].replace(
        {'Healthy': 0, 'Diabetes': 1, 'Thalasse': 2, 'Anemia': 3, 'Thromboc': 4})
    trainingDataFeatures = trainingData.drop('Disease', axis=1)

    return trainingDataFeatures, trainingLabels

def loadTestingData():
    testingData = pd.read_csv('blood_samples_dataset_test.csv')
    testingData = testingData[testingData['Disease'] != 'Heart Di']
    # shuffle the training data
    testingData.sample(frac=1)
    # replace the labels as numeric values
    testingLabels = testingData.iloc[:, -1].replace(
        {'Healthy': 0, 'Diabetes': 1, 'Thalasse': 2, 'Anemia': 3, 'Thromboc': 4})
    testDataFeatures = testingData.drop('Disease', axis=1)

    return testDataFeatures, testingLabels

def main():
    blood_features = getBloodFeatures()
    trainingData, trainingLabels = loadTrainingData()
    testingData, testingLabels = loadTestingData()

    # 70 % Training 30 % test
    X_train, X_test, y_train, y_test = train_test_split(trainingData, trainingLabels, test_size=0.3, random_state=42)

    treeClassifier = DecisionTreeClassifier(max_depth=4)
    # fit the decision model
    treeClassifier.fit(X_train, y_train)
    #  run classifier on the X test data
    y_predictions = treeClassifier.predict(X_test)


#      Training Data:
#    initial evaluation
    print("Accuracy:", metrics.accuracy_score(y_test, y_predictions))

    # Calculate the confusion matrix
    cm = confusion_matrix(y_test, y_predictions)

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(75, 30))
    plot_tree(treeClassifier, fontsize=15, filled=True, feature_names=blood_features, class_names=['Healthy', 'Diabetes', 'Thalasse', 'Anemia', 'Thromboc'])
    plt.show()

#    Testing Data
    y_testingPredicitons = treeClassifier.predict(testingData)

    #      Training Data:
    #    initial evaluation
    print("Accuracy:", metrics.accuracy_score(testingLabels, y_testingPredicitons))

    # Calculate the confusion matrix
    cm = confusion_matrix(testingLabels, y_testingPredicitons)

    # Plot the confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.show()

    plt.figure(figsize=(75, 30))
    plot_tree(treeClassifier, fontsize=15, filled=True, feature_names=blood_features,
              class_names=['Healthy', 'Diabetes', 'Thalasse', 'Anemia', 'Thromboc'])
    plt.show()

main()