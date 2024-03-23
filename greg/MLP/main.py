import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def preProcessTrainingData():
    trainingData = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')
    # shuffle the training data
    trainingData.sample(frac=1)
    # replace the labels as numeric values
    trainingLabels = trainingData.iloc[:,-1].replace({'Healthy':0,'Diabetes':1,'Thalasse':2,'Anemia':3,'Thromboc':4})
    trainingDataFeatures = trainingData.drop('Disease', axis=1)


    return trainingDataFeatures, trainingLabels

def preProcessTestingData():
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
    # aquire data to experiment with
    trainingData, trainingTargets = preProcessTrainingData()
    testingData, testingTargets = preProcessTestingData()

    X_train, X_test, y_train, y_test = train_test_split(trainingData, trainingTargets, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(24, 2000), activation="relu", alpha=10, batch_size=250, learning_rate_init=0.001, max_iter=1000, random_state=42)
    mlp.fit(X_train,y_train)

    y_prediction = mlp.predict(X_test)


    accuracy = accuracy_score(y_test, y_prediction)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(y_test, y_prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.show()

    y_test_prediction = mlp.predict(testingData)


    accuracy = accuracy_score(testingTargets, y_test_prediction)
    print(f"Accuracy: {accuracy:.2f}")

    cm = confusion_matrix(testingTargets, y_test_prediction)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g')
    plt.xlabel('Predicted labels')
    plt.ylabel('Actual labels')
    plt.title('Confusion Matrix')
    plt.show()


main()