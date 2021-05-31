# File:        read_data.py
#
# Author:      Rohan Patel
#
# Date:        05/09/2018
#
# Description: This script simply reads the sms data from the SMSSpamCollection file, prints the total number of
#              sms messages in the dataset, and then individually prints the first 100 lines from the SMSSpamCollection
#              file. The purpose of this script is to simply give an initial idea of how the sms data is organized in 
#              then dataset.

from random import *

def import_messages():
    # messages = [line.rstrip() for line in open('dataset/SMSSpamCollection')]
    messages = [line.rstrip() for line in open('regression_dataset/SMSSpamCollection_diff')]

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

        f.write(f'{label}\t{msg}\n')

    f.close()

def create_random_drift():
    # Data drift by randomness
    f = open("dataset/drifts/drift_random.txt", "w")
    messages = import_messages()

    for msg in messages:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]

        if random() > 0.5:
            if label == "spam":
                label = "ham"
            elif label == "ham":
                label = "spam"

        f.write(f'{label}\t{msg}\n')

    f.close()

def create_drift_mutation():
    # Data drift by mutation of ham messages by inserting new words from the spam messages
    f = open("dataset/drifts/drift_mutation.txt", "w")
    messages = import_messages()

    for msg in messages[:100]:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]




        f.write(f'{label}\t{msg}\n')

    f.close()


def create_drift_concept():
    # Concept drift by reducing training size and splitting the dataset (see paper on 
    # Concept drift for emails, not entirely sure how it works)
    f = open("dataset/drifts/drift_concept.txt", "w")
    messages = import_messages()

    for msg in messages[:100]:
        splitted = msg.split("\t")
        label = splitted[0]
        msg = splitted[1]




        f.write(f'{label}\t{msg}\n')

    f.close()


if __name__ == "__main__":
    create_drift_flip()
    create_random_drift()
    create_drift_mutation()
    create_drift_concept()

# Data detection using https://www.explorium.ai/blog/understanding-and-handling-data-and-concept-drift/

