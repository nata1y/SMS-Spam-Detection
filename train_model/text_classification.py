"""
Train the model using different algorithms.
Creates 3 files in output: `accuracy_scores.png`,
`model.joblib`, and `misclassified_msgs.txt`.
"""
import json
import random
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt

from joblib import dump, load
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from deploy_model.util import ensure_path_exists
from train_model.nlp_model import NLPModel
from train_model.util import OUTPUT_DIR, DATASET_DIR, load_data

pd.set_option('display.max_colwidth', None)
ensure_path_exists('output')


def my_train_test_split(*datasets):
    '''Split dataset into training and test sets. We use a 70/30 split.'''
    return train_test_split(*datasets, test_size=0.3, random_state=101)

def train_classifier(classifier, x_train, y_train):
    '''Traom classifiers with training set.'''
    classifier.fit(x_train, y_train)

def save_classifier(classifier, name):
    '''Save classifier to file.'''
    dump(classifier, OUTPUT_DIR + '{}_{}.joblib'.format(name,
        datetime.now().strftime("%m-%d-%Y")), compress=0, protocol=4)

def predict_labels(classifier, x_test):
    '''Predict the labels using the classifier.'''
    return classifier.predict(x_test)

def get_losses(losses, amount_subsamples):
    '''Sample subsets of the losses to calculate loss distribution.'''
    res = []
    for _ in range(amount_subsamples):
        loss_samples = random.sample(losses, 80)
        res.append(sum(loss_samples) / len(loss_samples))
    return res

def save_misclassified(classifiers, pred, pred_scores, datasets, test_messages):
    '''Save misclassified messages'''
    with open('output/misclassified_msgs.txt', 'a', encoding='utf-8') as file:
        for key, value in classifiers.items():
            train_classifier(value, datasets['x_train'], datasets['y_train'])
            pred[key] = predict_labels(value, datasets['x_test'])
            pred_scores[key] = [accuracy_score(datasets['y_test'], pred[key])]
            print('\n############### ' + key + ' ###############\n')
            print(classification_report(datasets['y_test'], pred[key]))
            write_misclassified(file, pred, key, test_messages, datasets['y_test'])

def write_misclassified(file, pred, key, test_messages, y_test):
    '''Write misclassified messages into a new text file.'''
    file.write('\n#################### ' + key + ' ####################\n')
    file.write('\nMisclassified Spam:\n\n')
    for msg in test_messages[y_test < pred[key]]:
        file.write(msg)
        file.write('\n')
    file.write('\nMisclassified Ham:\n\n')
    for msg in test_messages[y_test > pred[key]]:
        file.write(msg)
        file.write('\n')

def main():
    '''Trains multiple classifiers and stores the best one.'''
    raw_data = load_data(DATASET_DIR + 'SMSSpamCollection')
    NLPModel().train_nlp_model(raw_data)
    preprocessed_data = load('output/preprocessed_data.joblib')

    (x_train, x_test, y_train, y_test, _, test_messages) = my_train_test_split(
        preprocessed_data, raw_data['label'], raw_data['message'])

    classifiers = {
        'SVM': SVC(),
        'Decision Tree': DecisionTreeClassifier(),
        'Multinomial NB': MultinomialNB(),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(),
        'AdaBoost': AdaBoostClassifier(),
        'Bagging Classifier': BaggingClassifier()
    }

    pred_scores = dict()
    pred = dict()
    save_misclassified(classifiers, pred, pred_scores,
        {'x_train': x_train, 'y_train': y_train, 'x_test': x_test, 'y_test': y_test}, test_messages)

    print('\n############### Accuracy Scores ###############')
    accuracy = pd.DataFrame.from_dict(pred_scores, orient='index', columns=['Accuracy Rate'])
    print('\n')
    print(accuracy)
    print('\n')

    #plot accuracy scores in a bar plot
    accuracy.plot(kind='bar', ylim=(0.85, 1.0), edgecolor='black', figsize=(10, 5))
    plt.ylabel('Accuracy Score')
    plt.title('Distribution by Classifier')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("output/accuracy_scores.png")

    # Store "best" classifier
    dump(classifiers['Decision Tree'], 'output/model.joblib')

    best_clf = accuracy['Accuracy Rate'].idxmax()
    losses = get_losses([0.0 if y_test.tolist()[i] == pred[best_clf][i]
                            else 1.0 for i in range(len(y_test))], 100)
    losses = {'losses': losses}
    with open('output/losses.json', 'w') as file:
        json.dump(losses, file)

    save_classifier(classifiers[best_clf], best_clf.lower().replace(' ', '-'))

if __name__ == "__main__":
    main()
