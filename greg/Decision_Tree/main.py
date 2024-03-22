import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def loadTrainingData():
    trainingDataFeatures = pd.readcsv('Blood_spam)

def loadTestingData():
