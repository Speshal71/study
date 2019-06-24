import time
import numpy as np
import pandas as pd
from MLalgos import SVM, PolynomialRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, \
                            classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import CountVectorizer


def test_polynomial_regression():
    df = pd.read_csv('kc_house_data.csv')

    prices = df['price'].values
    features = df.drop(columns=['id', 'date', 'price']).values

    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        prices, 
                                                        test_size=0.3)
    
    for i in range(1, 4):
        start_time = time.perf_counter_ns()
        prd = PolynomialRegression(i)
        prd.fit(X_train, y_train)
        y_pred = prd.predict(X_test)
        end_time = time.perf_counter_ns()
        
        print('Polynomial Regression of {} order'.format(i))
        print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred))
        print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)))
        print('Time: ', (end_time - start_time) / 10 ** 9)
        print()
        

def test_classifier(clf, X_train, y_train, X_test, y_test):
    start_time = time.perf_counter_ns()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    end_time = time.perf_counter_ns()
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print('Time: ', (end_time - start_time) / 10 ** 9)
    print()


def test_svm_houses():
    df = pd.read_csv('kc_house_data.csv')
    
    elite = (df['price'] > 700000).map({True: 1, False: -1}).values
    features = df.drop(columns=['id', 'date', 'price']).values
    
    features = StandardScaler().fit_transform(features)
    X_train, X_test, y_train, y_test = train_test_split(features, 
                                                        elite, 
                                                        test_size=0.3)
    
    print('sklearn SVM:')
    test_classifier(SVC(gamma='auto'), X_train, y_train, X_test, y_test)
    print('custom SVM:')
    test_classifier(SVM(), X_train, y_train, X_test, y_test)


def test_svm_text():
    df = pd.read_csv('spam_or_not_spam.csv').dropna()
    
    spam = df['label'].map({0: -1, 1: 1}).values
    emails = df['email'].values
    
    vectorizer = CountVectorizer()
    vectorizer.fit(np.random.choice(emails, 100))
    
    emails_features = vectorizer.transform(emails).toarray()
    X_train, X_test, y_train, y_test = train_test_split(emails_features, 
                                                        spam, 
                                                        test_size=0.25)
    
    print('sklearn SVM:')
    test_classifier(SVC(gamma='auto'), X_train, y_train, X_test, y_test)
    print('custom SVM:')
    test_classifier(SVM(), X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    print('__________Polynomial Regression Test__________')
    test_polynomial_regression()
    print('__________SVM Houses Prediction__________')
    test_svm_houses()
    print('__________SVM Spam Prediction__________')
    test_svm_text()