import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from utilities import obtain_model, downsampling, obtain_data, time_based_splitting


GOOGLE_OUTPUT_FILE = r'confusion_google_'
BACKBLAZE_OUTPUT_FILE = r'confusion_disk_'
N_ROUNDS = 100
MODEL_NAME = ''
test_dic = {
    0.5: 'Random 50%/50%',
    0.4: 'Random 60%/40%',
    0.3: 'Random 70%/30%',
    0.2: 'Random 80%/20%',
    0.1: 'Random 90%/10%',
    -0.5: 'Time-based 50%/50%',
    -0.4: 'Time-based 60%/40%',
    -0.3: 'Time-based 70%/30%',
    -0.2: 'Time-based 80%/20%',
    -0.1: 'Time-based 90%/10%',
    0.0: 'Oracle'
}


def obtain_confusion(labels, probas):
    preds = probas > 0.5
    TN, FP, FN, TP = confusion_matrix(labels, preds).ravel()
    return [TP, TN, FP, FN]


def single_run(seen_features, unseen_features, seen_labels, unseen_labels, ratio):
    '''
    Build the model and evaluate the performance once for a specific data splitting technique

    Args:
        ratio < 0: time-based with testing set = (-ratio)
        ratio > 0: random sampling with testing set = ratio
        ratio = 0: oracle, trained on whole seen data and test on unseen data
    Returns:
        (list): performance metrics and feature importance in this run
    '''
    print('Spliting data')
    if np.isclose(ratio, 0):  # oracle
        training_features = seen_features
        testing_features = unseen_features
        training_labels = seen_labels
        testing_labels = unseen_labels
    elif ratio < 0:  # time-based splitting
        training_features, testing_features, training_labels, testing_labels = time_based_splitting(seen_features, seen_labels, 1 + ratio)
    else:  # random splitting
        training_features, testing_features, training_labels, testing_labels = train_test_split(seen_features, seen_labels, test_size = ratio)

    print(np.count_nonzero(training_labels)/len(training_labels))
    # remove time column, scaling and downsampling
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features[:, 1:])
    testing_features = scaler.transform(testing_features[:, 1:])
    unseen_features = scaler.transform(unseen_features[:, 1:])
    training_features, training_labels = downsampling(training_features, training_labels)

    print('Training with features', training_features.shape)
    model = obtain_model(MODEL_NAME)
    model.fit(training_features, training_labels)

    print('Start prediction')
    testing_probas = model.predict_proba(testing_features)[:, 1]
    unseen_probas = model.predict_proba(unseen_features)[:, 1]

    # obtain evaluation results
    print('Testing features', testing_labels.shape[0], np.count_nonzero(testing_labels))
    print('Oracle features', unseen_labels.shape[0], np.count_nonzero(unseen_labels))
    results = [[test_dic[ratio], 'Testing'] + obtain_confusion(testing_labels, testing_probas),
               [test_dic[ratio], 'Oracle'] + obtain_confusion(unseen_labels, unseen_probas)]

    return results


def experiment_driver(features, labels, output_file):
    '''
    '''
    out_columns = ['Test Name', 'Scenario', 'TP', 'TN', 'FP', 'FN']

    seen_features, unseen_features, seen_labels, unseen_labels = time_based_splitting(features, labels, 0.5)
    print('Seen data shape:', seen_features.shape)
    print('Unseen data shape:', unseen_features.shape)

    ratios = [0.3, -0.3]
    for ratio in ratios:
        print('Test objectives:', test_dic[ratio])
        for i in range(N_ROUNDS):
            print('Start round #', i)
            output = single_run(seen_features, unseen_features, seen_labels, unseen_labels, ratio)
            if i == 0:  # Print performance for the first run
                print(output)

            out_df = pd.DataFrame(output, columns=out_columns)
            out_df.to_csv(output_file, mode='a', index=False, header=(not os.path.isfile(output_file)))
        print()


if __name__ == '__main__':
    MODEL_CHOICE = ['rf', 'nn', 'cart', 'rgf', 'svm']
    parser = argparse.ArgumentParser(description='Experiment on time-based splitting vs. random splitting')
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-m", help="specify the model, random forest by default.", default='rf', choices=MODEL_CHOICE)
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    #parser.add_argument('-a', action='store_true', help="Specify -a argument to run all five models at once, one by one.")
    args = parser.parse_args()

    MODEL_NAME = args.m
    N_ROUNDS = int(args.n)

    if args.d == 'g':
        print('Choose Google as dataset')
        OUTPUT_FILE = GOOGLE_OUTPUT_FILE + args.m + '.csv'
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_FILE + args.m + '.csv'
    else:
        exit(-1)

    if os.path.isfile(OUTPUT_FILE): 
        os.remove(OUTPUT_FILE)
    print('Output path:', OUTPUT_FILE)

    features, labels = obtain_data(args.d, 'd')
    experiment_driver(features, labels, OUTPUT_FILE)
        
    print('Experiment completed!')
