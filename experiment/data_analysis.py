from sklearn.preprocessing import StandardScaler
from utilities import *
import argparse
from scipy import stats
from sklearn.model_selection import KFold


data_names = {
    'g': 'google',
    'b': 'disk'
}


def analyze_failure_rate(dataset):
    features, labels = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)

    x = []
    y = []
    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        testing_labels = labels[idx]
        if len(testing_labels) < 1:
            continue
            
        rate = np.count_nonzero(testing_labels == True) / len(testing_labels)
        y.append(rate)
        x.append(i+1)
        
    df = pd.DataFrame({'x': x, 'y': y})
    df.to_csv('failure_rate_' + data_names[dataset] + '.csv', index=False)


def cross_validation(model, features, labels):
    kf = KFold(n_splits=10, shuffle=False)
    error_num = 0
    total_num = 0
    for training_index, testing_index in kf.split(features):
        training_features, training_labels = features[training_index], labels[training_index]
        testing_features, testing_labels = features[testing_index], labels[testing_index]
        model.fit(training_features, training_labels)
        testing_preds = model.predict(testing_features)
        error_num += np.count_nonzero(testing_preds != testing_labels)
        total_num += len(testing_labels)

    return error_num, total_num


def analyze_concept_drift(dataset, model_name='rf'):
    features, labels = obtain_data(dataset, 'm')
    scaler = StandardScaler()
    features[:, 1:] = scaler.fit_transform(features[:, 1:])
    feature_list, label_list = obtain_natural_chunks(features, labels, obtain_intervals(dataset))
    out_columns = ['Training Period', 'Testing Period', 'Training Error', 'Testing Error', 'Training Size', 'Testing Size']
    out_ls = []

    for _ in range(10):
        for i in range(len(feature_list) - 1):
            print('On period', i+1)
            training_features = feature_list[i]
            training_labels = label_list[i]
            
            model = obtain_model(model_name)
            training_err, training_len = cross_validation(obtain_model(model_name), training_features, training_labels)
            model.fit(training_features, training_labels)

            testing_features = feature_list[i + 1]
            testing_labels = label_list[i + 1]
            testing_preds = model.predict(testing_features)
            testing_err = np.count_nonzero(testing_labels != testing_preds)
            out_ls.append([i + 1, i + 2, training_err, testing_err, training_len, len(testing_labels)])
            
        out_df = pd.DataFrame(out_ls, columns=out_columns)
        out_df.to_csv('concept_drift_' + data_names[dataset] + '_' + model_name + '.csv', index=False)


def analyze_explanatory_corr(dataset):
    features, _ = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)
    scaler = StandardScaler()
    features[:, 1:] = scaler.fit_transform(features[:, 1:])

    columns = ['Timestamp'] + obtain_feature_names(dataset)
    pairs = []
    for i in range(1, features.shape[1]):
        for j in range(i + 1, features.shape[1]):
            corrs = []
            for n in range(len(terminals) - 1):
                idx = np.logical_and(features[:, 0] >= terminals[n], features[:, 0] < terminals[n + 1])
                corr, p = stats.spearmanr(features[idx][:, i], features[idx][:, j])
                corrs.append(corr)
            pairs.append((np.std(corrs), i, j, columns[i]+'~'+columns[j]))
    pairs = np.array(pairs, dtype=[('std', float), ('i', int), ('j', int), ('index', object)])
    pairs = np.sort(pairs, order=['std'])[::-1]

    print(pairs)

    out_ls = []
    for n in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[n], features[:, 0] < terminals[n + 1])
        sample_features = features[idx]
        for idx in range(10):
            i = pairs[idx][1]
            j = pairs[idx][2]
            corr, p = stats.spearmanr(sample_features[:, i], sample_features[:, j])
            out_ls.append([n + 1, str(idx+1)+'-'+pairs[idx][3], corr, p])
            
    df = pd.DataFrame(out_ls, columns=['Period', 'Index', 'Corr', 'P Value'])
    df.to_csv('explanatory_corr_' + data_names[dataset] + '.csv', index=False)
        

def analyze_target_corr(dataset):
    features, labels = obtain_data(dataset, 'm')
    terminals = obtain_intervals(dataset)
    scaler = StandardScaler()
    features[:, 1:] = scaler.fit_transform(features[:, 1:])

    columns = ['Timestamp'] + obtain_feature_names(dataset)
    pairs = []
    for i in range(1, features.shape[1]):
        corrs = []
        for n in range(len(terminals) - 1):
            idx = np.logical_and(features[:, 0] >= terminals[n], features[:, 0] < terminals[n + 1])
            corr, p = stats.spearmanr(features[idx][:, i], labels[idx])
            corrs.append(corr)
        pairs.append((np.std(corrs), i, columns[i]))
    pairs = np.array(pairs, dtype=[('std', float), ('i', int), ('index', object)])
    pairs = np.sort(pairs, order=['std'])[::-1]

    out_ls = []
    for n in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[n], features[:, 0] < terminals[n + 1])
        sample_features = features[idx]
        sample_labels = labels[idx]
        for idx in range(10):
            i = pairs[idx][1]
            corr, p = stats.spearmanr(sample_features[:, i], sample_labels)
            out_ls.append([n + 1, str(idx+1)+'-Target~'+pairs[idx][2], corr, p])
            
    df = pd.DataFrame(out_ls, columns=['Period', 'Index', 'Corr', 'P Value'])
    df.to_csv('target_corr_' + data_names[dataset] + '.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiment on static vs. updated models')
    parser.add_argument("-m", help="specify the model, random forest by default.", default='rf', choices=['rf', 'nn', 'cart', 'rgf', 'svm'])
    args = parser.parse_args()

    analyze_failure_rate('g')
    analyze_failure_rate('b')

    analyze_concept_drift('g', args.m)
    analyze_concept_drift('b', args.m)

    analyze_explanatory_corr('g')
    analyze_explanatory_corr('b')

    analyze_target_corr('g')
    analyze_target_corr('b')
    
    print('Experiment completed!')
