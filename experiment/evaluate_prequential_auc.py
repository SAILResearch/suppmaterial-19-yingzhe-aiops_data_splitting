import os
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from utilities import obtain_model, obtain_data, downsampling, obtain_intervals, obtain_natural_chunks


GOOGLE_OUTPUT_FILE = r'prequential_google_'
BACKBLAZE_OUTPUT_FILE = r'prequential_disk_'
N_ROUNDS = 100
MODEL_NAME = ''


def experiment_driver(feature_list, label_list, out_file):
    '''
    '''
    out_columns = ['Period', 'Static', 'Update', 'Label']
    out_ls = []

    num_chunks = len(feature_list)
    print('Total number of periods:', num_chunks)
    print('Building models from first', num_chunks//2, 'periods')

    # obtain static training features and labels
    training_features = np.vstack(feature_list[0: num_chunks//2])
    training_labels = np.hstack(label_list[0: num_chunks//2])
    # scaler and downsampling for static model
    static_scaler = StandardScaler()
    training_features = static_scaler.fit_transform(training_features)
    training_features, training_labels = downsampling(training_features, training_labels)

    print('Build static model')
    static_model = obtain_model(MODEL_NAME)
    static_model.fit(training_features, training_labels)

    for i in range(num_chunks//2, num_chunks):
        # obtain sliding-window training features and labels
        training_features = np.vstack(feature_list[i - num_chunks//2: i])
        training_labels = np.hstack(label_list[i - num_chunks//2: i])
        # scaler and downsampling for sliding-window model
        update_scaler = StandardScaler()
        training_features = update_scaler.fit_transform(training_features)
        training_features, training_labels = downsampling(training_features, training_labels)

        # obtain testing features and labels
        testing_features = feature_list[i]
        testing_labels = label_list[i]

        print('Building update model')
        update_model = obtain_model(MODEL_NAME)
        update_model.fit(training_features, training_labels)
        
        print('Testing models on period', i + 1)
        static_probas = static_model.predict_proba(static_scaler.transform(testing_features))[:, 1]
        update_probas = update_model.predict_proba(update_scaler.transform(testing_features))[:, 1]

        #output = np.vstack((static_probas, update_probas, testing_labels.astype(object))).reshape(-1, 3)
        output = [(i+1, static_probas[idx], update_probas[idx], testing_labels[idx]) for idx in range(len(testing_labels))]
        
        out_df = pd.DataFrame(output, columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment on static vs. updated models')
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    args = parser.parse_args()

    N_ROUNDS = int(args.n)
    features, labels = obtain_data(args.d, 'm')
    terminals = obtain_intervals(args.d)
    feature_list, label_list = obtain_natural_chunks(features, labels, terminals)

    if args.d == 'g':
        print('Choose Google as dataset')
        OUTPUT_FILE = GOOGLE_OUTPUT_FILE
    elif args.d == 'b':
        print('Choose Backblaze as dataset')
        OUTPUT_FILE = BACKBLAZE_OUTPUT_FILE
    else:
        exit(-1)

    for i in range(N_ROUNDS):
        for m in ['rf', 'nn', 'cart', 'svm', 'rgf']:
            MODEL_NAME = m
            
            OUTPUT_FILE_FIX = OUTPUT_FILE + m + '_' + str(i) + '.csv'

            if os.path.isfile(OUTPUT_FILE_FIX): 
                os.remove(OUTPUT_FILE_FIX)
            print('Output path:', OUTPUT_FILE_FIX)
            experiment_driver(feature_list, label_list, OUTPUT_FILE_FIX)
        
    print('Experiment completed!')
