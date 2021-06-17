# pylint: disable=R0801
'''Generate script for different types of drifts.'''
import string
import random
import nltk

from nltk.tokenize import word_tokenize

from deploy_model.util import ensure_path_exists

nltk.download('punkt')
ensure_path_exists('dataset/drifts')


def import_messages():
    '''Import the messages from the training dataset.'''
    return [line.rstrip() for line in open('dataset/SMSSpamCollection')]

def clean(text):
    '''Clean up a piece of text.'''
    tokens = word_tokenize(text)
    tokens = [words.lower() for words in tokens] # convert to lower case
    table = str.maketrans('', '', string.punctuation) # remove punctuation
    stripped = [w.translate(table) for w in tokens] # remove remaining tokens
    words = [word for word in stripped if word.isalpha()]
    # stop_words = set(stopwords.words('english')) # filter out stop words
    # words = [w for w in words if not w in stop_words]
    return words

def create_drift_flip():
    '''Data drift by flipping the labels.'''
    with open("dataset/drifts/drift_flip.txt", "w") as file:
        messages = import_messages()
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if label == "spam":
                label = "ham"
            elif label == "ham":
                label = "spam"
            file.write(f'{label}\t{msg}\n')

def create_random_drift(probability):
    '''Data drift by randomly changing labels.'''
    with open("dataset/drifts/drift_random_" + str(probability) + ".txt", "w") as file:
        messages = import_messages()
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if random.random() > probability:
                if label == "spam":
                    label = "ham"
                elif label == "ham":
                    label = "spam"
            file.write(f'{label}\t{msg}\n')

def create_drift_mutation():
    '''Data drift by mutation of ham messages by inserting new words from the spam messages.'''
    with open("dataset/drifts/drift_mutation.txt", "w") as file:
        messages = import_messages()
        total_words = 0
        word_dict = {}
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if label == "spam":
                nltk_tokens = clean(msg)
                for word in nltk_tokens:
                    if word in word_dict:
                        word_dict[word] += 1
                    else:
                        word_dict[word] = 0
                    total_words += len(nltk_tokens)
        top_worst = []
        count = 0
        for words in sorted(word_dict, key=word_dict.get, reverse=True):
            if len(words) > 3 and word_dict[words] > 0 and count < 20:
                top_worst.append(words)
                count += 1
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if label == "ham":
                for _ in range(0, 5):
                    msg += " " + random.choice(top_worst)
            file.write(f'{label}\t{msg}\n')

def create_drift_concept():
    '''Concept drift by reducing training size and splitting the dataset (see paper on
    Concept drift for emails) previous papers used this approach.'''
    with open("dataset/drifts/drift_concept.txt", "w") as file:
        messages = import_messages()
        for msg in messages[:int(len(messages)/2)]:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]

            file.write(f'{label}\t{msg}\n')

def create_drift_spam():
    '''Only have spam in data, introducing ham later on can possibly cause a drift.'''
    with open("dataset/drifts/drift_spam_only.txt", "w") as file:
        messages = import_messages()
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if label == "spam":
                file.write(f'{label}\t{msg}\n')

def create_drift_ham():
    '''Vice versa to create_drift_spam.'''
    with open("dataset/drifts/drift_ham_only.txt", "w") as file:
        messages = import_messages()
        for msg in messages:
            splitted = msg.split("\t")
            label = splitted[0]
            msg = splitted[1]
            if label == "ham":
                file.write(f'{label}\t{msg}\n')

def generate_all_drifts():
    '''Generate all types of drifts based on the dataset.'''
    create_drift_flip()
    create_random_drift(0.5)
    create_drift_mutation()
    create_drift_concept()
    create_drift_spam()
    create_drift_ham()

if __name__ == "__main__":
    generate_all_drifts()
