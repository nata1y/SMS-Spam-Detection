import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

def SVM_Tuning(X_train, X_test, y_train, y_test):

	print('\n############### SVM ###############\n')
	param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}

	model = GridSearchCV(SVC(kernel = 'sigmoid'), param_grid, verbose = 1)
	model.fit(X_train, y_train)

	print('\nBest parameter:', model.best_params_)

	pred = model.predict(X_test)

	print('\nAccuracy Score:', accuracy_score(y_test, pred))
	print('\n')
	print(classification_report(y_test, pred))

def MNB_Tuning(X_train, X_test, y_train, y_test):

	print('\n############### SVM ###############\n')
	param_grid = {'alpha': np.arange(0.0, 1.05, 0.05)}

	model = GridSearchCV(MultinomialNB(), param_grid, verbose = 1)
	model.fit(X_train, y_train)

	print('\nBest parameter:', model.best_params_)

	pred = model.predict(X_test)

	print('\nAccuracy Score:', accuracy_score(y_test, pred))
	print('\n')
	print(classification_report(y_test, pred))

def main():

	tfidf_vect = pickle.load(open("output/tfidf_vector.pickle", "rb"))
	messages = pd.read_csv('output/processed_msgs.csv')

	X_train, X_test, y_train, y_test = train_test_split(tfidf_vect, messages['label'], test_size = 0.3, random_state = 101)

	SVM_Tuning(X_train, X_test, y_train, y_test)
	MNB_Tuning(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()