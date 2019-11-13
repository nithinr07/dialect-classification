import warnings
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb

class Dialect_Classifier:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)
        dialect_list = self.data.iloc[:, -1]
        encoder = LabelEncoder()
        self.labels = encoder.fit_transform(dialect_list)
        self.data1 = self.data.iloc[:,:20]
        self.data2 = self.data1
        self.data2['spectral_flux'] = self.data['spectral_flux']        
        scaler = StandardScaler()

        # Uncomment at a time one of the lines to check the performances of the dataframes as mentioned in the report(in the same order as in the report)
        # Current dataframe consists of MFCC + Spectral Flux features.

        # self.X = scaler.fit_transform(np.array(self.data.iloc[:,:-1], dtype = float))
        # self.X = scaler.fit_transform(np.array(self.data1, dtype = float))
        self.X = scaler.fit_transform(np.array(self.data2, dtype = float))
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.labels, test_size=0.2)

    def logistic_classifier(self):
        logistic_classifier = linear_model.logistic.LogisticRegression()
        logistic_classifier.fit(self.X_train, self.y_train)
        logistic_predictions = logistic_classifier.predict(self.X_test)
        logistic_accuracy = accuracy_score(self.y_test, logistic_predictions)
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        cv_results = cross_val_score(logistic_classifier, self.X, self.labels, cv=kfold)
        logistic_cm = confusion_matrix(self.y_test, logistic_predictions)
        print("logistic accuracy = " + str(logistic_accuracy))
        print("average = " + str(np.mean(cv_results)) + "\nstd dev = " + str(np.std(cv_results)))
        print(logistic_cm)
        print(classification_report(self.y_test, logistic_predictions))

    def svm_classifier(self):
        svm_classifier = svm.SVC()
        svm_classifier.fit(self.X_train, self.y_train)
        svm_predictions = svm_classifier.predict(self.X_test)
        svm_accuracy = accuracy_score(self.y_test, svm_predictions)
        svm_cm = confusion_matrix(self.y_test, svm_predictions)
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        cv_results = cross_val_score(svm_classifier, self.X, self.labels, cv=kfold)
        print("svm accuracy = " + str(svm_accuracy))
        print("average = " + str(np.mean(cv_results)) + "\nstd dev = " + str(np.std(cv_results)))
        print(svm_cm)
        print(classification_report(self.y_test, svm_predictions))

    def knn_classifier(self, n):
        knn_classifier = KNeighborsClassifier(n_neighbors = n)
        knn_classifier.fit(self.X_train, self.y_train)
        knn_predictions = knn_classifier.predict(self.X_test)
        knn_accuracy = accuracy_score(self.y_test, knn_predictions)
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        cv_results = cross_val_score(knn_classifier, self.X, self.labels, cv=kfold)
        knn_cm = confusion_matrix(self.y_test, knn_predictions)
        print("knn accuracy = " + str(knn_accuracy))
        print("average = " + str(np.mean(cv_results)) + "\nstd dev = " + str(np.std(cv_results)))
        print(knn_cm)
        print(classification_report(self.y_test, knn_predictions))
        
    def random_forest_classifier(self):
        rf_classifier = RandomForestClassifier(n_estimators = 1000, random_state = 42)
        rf_classifier.fit(self.X_train, self.y_train)
        rf_predictions = rf_classifier.predict(self.X_test)
        rf_accuracy = accuracy_score(self.y_test, rf_predictions)
        kfold = KFold(n_splits=10, shuffle=True, random_state=5)
        cv_results = cross_val_score(rf_classifier, self.X, self.labels, cv=kfold)
        rf_cm = confusion_matrix(self.y_test, rf_predictions)
        print("rf accuracy = " + str(rf_accuracy))
        print("average = " + str(np.mean(cv_results)) + "\nstd dev = " + str(np.std(cv_results)))
        print(rf_cm)
        print(classification_report(self.y_test, rf_predictions)) 

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    dialect_classifier = Dialect_Classifier('data.csv')
    dialect_classifier.logistic_classifier()
    dialect_classifier.svm_classifier()
    dialect_classifier.knn_classifier(3)
    dialect_classifier.knn_classifier(5)
    dialect_classifier.random_forest_classifier()