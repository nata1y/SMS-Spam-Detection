import nltk
import string
import random
from nltk.tokenize import word_tokenize

from deploy_model.util import ensure_path_exists

nltk.download('punkt')
ensure_path_exists('dataset/drifts')


def import_messages():
    messages = [line.rstrip() for line in open('dataset/SMSSpamCollection')]
    # messages = [line.rstrip() for line in open('regression_dataset/SMSSpamCollection_diff')]

    # print('Total number of messages: ' + str(len(messages)))
    return messages


def create_drift_flip():
    # Data drift by flipping
    f = open("dataset/drifts/drift_flip.txt", "w")
    messages = import_messages()

    for msg in messages:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        if label == "spam":
            label = "ham"
        elif label == "ham":
            label = "spam"

        f.write(f"{label}\t{msg}\n")

    f.close()


def create_random_drift(probability):
    # Data drift by randomness
    f = open("dataset/drifts/drift_random_" + str(probability) + ".txt", "w")
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

        f.write(f'{label}\t{msg}\n')

    f.close()


def clean(text):
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each wor
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    # stop_words = set(stopwords.words('english'))
    # words = [w for w in words if not w in stop_words]

    return words


def create_drift_mutation():
	# Data drift by mutation of ham messages by inserting new words from the spam messages
    f = open("dataset/drifts/drift_mutation.txt", "w")
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
    for w in sorted(word_dict, key=word_dict.get, reverse=True):
        if len(w) > 3 and word_dict[w] > 0 and count < 20:
            top_worst.append(w)
            count += 1

    for msg in messages:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        if label == "ham":
            for i in range(0, 5):
                msg += " " + random.choice(top_worst)

        f.write(f'{label}\t{msg}\n')

    f.close()


def create_drift_concept():
    # Concept drift by reducing training size and splitting the dataset (see paper on 
    # Concept drift for emails) previous papers used this approach
    f = open("dataset/drifts/drift_concept.txt", "w")
    messages = import_messages()
    for msg in messages[:int(len(messages)/2)]:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        f.write(f'{label}\t{msg}\n')

    f.close()


def create_drift_spam():
	# only have spam in data, introducing ham later on can possibly cause a drift.
    f = open("dataset/drifts/drift_spam_only.txt", "w")
    messages = import_messages()

    for msg in messages:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        if label == "spam":
        	f.write(f'{label}\t{msg}\n')

    f.close()


def create_drift_ham():
	# vice versa to create_drift_spam
    f = open("dataset/drifts/drift_ham_only.txt", "w")
    messages = import_messages()

    for msg in messages:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        if label == "ham":
        	f.write(f'{label}\t{msg}\n')

    f.close()


if __name__ == "__main__":
    create_drift_flip()
    create_random_drift(0.5)
    create_drift_mutation()
    create_drift_concept()
    create_drift_spam()
    create_drift_ham()

# Data detection using https://www.explorium.ai/blog/understanding-and-handling-data-and-concept-drift/

