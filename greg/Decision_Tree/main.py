import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

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
    # shuffle the training data
    testingData.sample(frac=1)
    # replace the labels as numeric values
    testingLabels = testingData.iloc[:, -1].replace(
        {'Healthy': 0, 'Diabetes': 1, 'Thalasse': 2, 'Anemia': 3, 'Thromboc': 4})
    testDataFeatures = testingData.drop('Disease', axis=1)

    return testDataFeatures, testingLabels

def main():


main()