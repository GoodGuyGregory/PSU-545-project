import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

def preprocess():
    data = pd.read_csv('Blood_samples_dataset_balanced_2(f).csv')

    ## replacing the values inside column disease with values
    data['Disease'].replace({'Anemia': 0, 'Healthy':1, 'Diabetes':2, 'Thalasse': 3, 'Thromboc': 4}, inplace=True)

    ## separating the features from the targets
    x = data.iloc[:,0:24]
    y = data.iloc[:,-1:]

    ## performing train test split
    x_train, x_test, y_train, y_test = train_test_split(x,y ,test_size= 0.3, random_state= 34)
    return x_train, x_test, y_train, y_test

def train_test(x_train,y_train,x_test,y_test):
    naive = GaussianNB()
    naive.fit(x_train, y_train.values.ravel())
    pred2 = naive.predict(x_test)

    ## accuracy of the model
    acc = metrics.accuracy_score(y_test,pred2)
    print(acc)

     ## classification  report of the model
    names = ['Anemia', 'Healthy', 'Diabetes', 'Thalasse', 'Thromboc']
    print(classification_report(y_test, pred2, target_names= names))

    confs = metrics.confusion_matrix(y_test, pred2)
    dis = metrics.ConfusionMatrixDisplay(confusion_matrix= confs, display_labels=['Anemia', 'Healthy', 'Diabetes', 'Thalasse', 'Thromboc'])
    dis.plot()
    plt.show()
    
x_train, x_test, y_train, y_test = preprocess()
train_test(x_train,y_train,x_test,y_test)