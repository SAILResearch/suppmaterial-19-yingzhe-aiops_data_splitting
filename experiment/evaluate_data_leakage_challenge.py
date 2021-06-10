from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utilities import *
import argparse


GOOGLE_OUTPUT_FILE = r'leakage_google_'
BACKBLAZE_OUTPUT_FILE = r'leakage_disk_'
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


def single_run(features, labels, ratio):
    '''
    Build the model and evaluate the performance once for a specific data splitting technique

    Args:
        ratio < 0: time-based with testing set = (-ratio)
        ratio > 0: random sampling with testing set = ratio
    Returns:
        (list): performance metrics and feature importance in this run
    '''
    print('Spliting data')
    if ratio < 0:  # time-based splitting
        training_features, testing_features, training_labels, testing_labels = time_based_splitting(features, labels, 1 + ratio)
    else:  # random splitting
        training_features, testing_features, training_labels, testing_labels = train_test_split(features, labels, test_size = ratio)

    # remove time column, scaling and downsampling
    scaler = StandardScaler()
    training_features = scaler.fit_transform(training_features[:, 1:])
    testing_features = scaler.transform(testing_features[:, 1:])
    training_features, training_labels = downsampling(training_features, training_labels)

    print('Training with features', training_features.shape)
    model = obtain_model(MODEL_NAME)
    model.fit(training_features, training_labels)

    print('Start prediction')
    testing_probas = model.predict_proba(testing_features)[:, 1]

    out_list = []
    testing_preds = testing_probas > 0.5
    out_list.append(['Time-based' if ratio < 0 else 'Random', 'AUC', metrics.roc_auc_score(testing_labels, testing_probas)])
    out_list.append(['Time-based' if ratio < 0 else 'Random', 'F-1', metrics.f1_score(testing_labels, testing_preds)])
    out_list.append(['Time-based' if ratio < 0 else 'Random', 'MCC', metrics.matthews_corrcoef(testing_labels, testing_preds)])
    return out_list


def experiment_driver(features, labels, out_file):
    '''
    '''
    out_columns = ['Splitting Approach', 'Metric', 'Performance Value']
    ratios = [-0.3, 0.3]
    for ratio in ratios:
        print('Test objectives:', test_dic[ratio])
        for i in range(N_ROUNDS):
            print('Start round #', i)
            output = single_run(features, labels, ratio)
            out_df = pd.DataFrame(output, columns=out_columns)
            out_df.to_csv(out_file, mode='a', index=False, header=(not os.path.isfile(out_file)))
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment on time-based splitting vs. random splitting')
    parser.add_argument("-d", help="specify the dataset, d for Googole and b for Backblaze.", required=True, choices=['g', 'b'])
    parser.add_argument("-m", help="specify the model, random forest by default.", default='rf', choices=['rf', 'nn', 'cart', 'rgf', 'lr', 'gbdt', 'svm'])
    parser.add_argument("-n", help="specify the testing rounds, 100 by default.", default=100)
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

    features, labels = obtain_data(args.d)
    experiment_driver(features, labels, OUTPUT_FILE)
        
    print('Experiment completed!')
