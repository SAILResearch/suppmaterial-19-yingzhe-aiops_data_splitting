from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from utilities import *
import argparse
import timeit


GOOGLE_OUTPUT_FILE = r'window_google_'
BACKBLAZE_OUTPUT_FILE = r'window_disk_'
N_ROUNDS = 100
MODEL_NAME = ''


def experiment_driver(feature_list, label_list, out_file):
    '''
    '''
    out_columns = ['Scenario', 'Model', 'N', 'K', 'Test P', 'Test R', 'Test A', 'Test F', 'Test AUC', 'Test MCC', 'Test B', 'Training Time', 'Testing Time']
    out_ls = []

    num_chunks = len(feature_list)

    # obtain static training features and labels
    training_features = np.vstack(feature_list[0: num_chunks//2])
    training_labels = np.hstack(label_list[0: num_chunks//2])
    # scaler and downsampling for static model
    static_scaler = StandardScaler()
    training_features = static_scaler.fit_transform(training_features)
    training_features, training_labels = downsampling(training_features, training_labels)
    
    print('Build static model')
    static_model = obtain_model(MODEL_NAME)
    start_time = timeit.default_timer()
    static_model.fit(training_features, training_labels)
    static_training_time = timeit.default_timer() - start_time

    static_proba_ls = []
    static_time_ls = []
    update_proba_ls = []
    update_time_ls = []
    testing_label_ls = []
    # i stand for the index of testing period
    for i in range(num_chunks//2, num_chunks):
        # process sliding window feature
        training_features = np.vstack(feature_list[i - num_chunks//2: i])
        training_labels = np.hstack(label_list[i - num_chunks//2: i])
        # scaler and downsampling for sliding-window model
        update_scaler = StandardScaler()
        training_features = update_scaler.fit_transform(training_features)
        training_features, training_labels = downsampling(training_features, training_labels)

        # obtain testing feature and label
        testing_features = feature_list[i]
        testing_labels = label_list[i]
        testing_label_ls.append(testing_labels)

        print('Building update model')
        update_model = obtain_model(MODEL_NAME)
        start_time = timeit.default_timer()
        update_model.fit(training_features, training_labels)
        update_training_time = timeit.default_timer() - start_time

        print('Testing models on period', i + 1)
        # test static model
        start_time = timeit.default_timer()
        static_probas = static_model.predict_proba(static_scaler.transform(testing_features))[:, 1]
        static_testing_time = timeit.default_timer() - start_time
        static_proba_ls.append(static_probas)
        static_time_ls.append([static_training_time, static_testing_time])
        out_ls.append(['Static Model', MODEL_NAME.upper(), num_chunks, i+1] + obtain_metrics(testing_labels, static_probas) + static_time_ls[-1])

        # test update model
        start_time = timeit.default_timer()
        update_probas = update_model.predict_proba(update_scaler.transform(testing_features))[:, 1]
        update_testing_time = timeit.default_timer() - start_time
        update_proba_ls.append(update_probas)
        update_time_ls.append([update_training_time, update_testing_time])
        out_ls.append(['Updated Model', MODEL_NAME.upper(), num_chunks, i+1] + obtain_metrics(testing_labels, update_probas) + update_time_ls[-1])

        out_df = pd.DataFrame(out_ls[-2:], columns=out_columns)
        out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))

    # test performance on all latter half of data
    testing_labels = np.hstack(testing_label_ls)
    static_probas = np.hstack(static_proba_ls)
    update_probas = np.hstack(update_proba_ls)
    out_ls.append(['Static Model', MODEL_NAME.upper(), num_chunks, -1] + obtain_metrics(testing_labels, static_probas) +  [np.mean(np.array(static_time_ls)[:, 0]), np.mean(np.array(static_time_ls)[:, 1])])
    out_ls.append(['Updated Model', MODEL_NAME.upper(), num_chunks, -1] + obtain_metrics(testing_labels, update_probas) +  [np.mean(np.array(update_time_ls)[:, 0]), np.mean(np.array(update_time_ls)[:, 1])])
    out_df = pd.DataFrame(out_ls[-2:], columns=out_columns)
    out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment on static vs. updated models')
    parser.add_argument("-m", help="specify the model, random forest by default.", default='rf', choices=['rf', 'nn', 'cart', 'rgf', 'svm'])
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
    args = parser.parse_args()

    N_ROUNDS = int(args.n)
    features, labels = obtain_data(args.d)
    features = features[:, 1:]  # remove the timestamp column

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

    MODEL_NAME = args.m
    for _ in range(N_ROUNDS):
        for n in range(4, 26, 2):
            feature_list, label_list = obtain_chunks(features, labels, n)
            experiment_driver(feature_list, label_list, OUTPUT_FILE)
        
    print('Experiment completed!')
