from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from rgf.sklearn import RGFClassifier
from sklearn.svm import SVC

from sklearn import metrics, preprocessing
from sklearn.utils import resample
import pandas as pd
import numpy as np
import traceback
import mkl
import os

mkl.set_num_threads(1)  # control the number of thread used for NN model
N_WORKERS = 1  # control the number of workers used for RF model

#INPUT_FOLDER = r'/home/local/SAIL/yingzhe/ops_data/Google_cluster_data/clusterdata-2011-2/google_job/'
INPUT_FOLDER = r'./'
GOOGLE_INPUT_FILE = r'google_job_failure.csv'
BACKBLAZE_INPUT_FILE = r'disk_failure.csv'


def obtain_data(dataset, interval='m', include_end_time=False):
    if dataset == 'g':
        return get_google_data(include_end_time)
    elif dataset == 'b':
        return get_disk_data(interval)


def obtain_feature_names(dataset):
    if dataset == 'g':
        return ['User ID', 'Job Name', 'Scheduling Class',
               'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
               'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    elif dataset == 'b':
        return ['smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
        'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff']


def get_google_data(include_end_time):
    '''
    Read the Google dataset from csv file
    The input path and filename are specified in macros
    Return features and labels after proper preprocessing

    Returns:
        features (np.array): feature vector, the first column is the timestamp
        labels (np.array): True or False, binary classification
    '''
    path = os.path.join(INPUT_FOLDER, GOOGLE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path)

    if include_end_time:
        columns = ['Start Time', 'End Time', 'User ID', 'Job Name', 'Scheduling Class',
                   'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
                   'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    else:
        columns = ['Start Time', 'User ID', 'Job Name', 'Scheduling Class',
                   'Num Tasks', 'Priority', 'Diff Machine', 'CPU Requested', 'Mem Requested', 'Disk Requested',
                   'Avg CPU', 'Avg Mem', 'Avg Disk', 'Std CPU', 'Std Mem', 'Std Disk']
    print('Load complete')

    features = df[columns].to_numpy()
    labels = (df['Status']==3).to_numpy()

    print('Preprocessing features')
    offset = (1 if include_end_time else 0)

    # encode user id
    le = preprocessing.LabelEncoder()
    features[:, 1+offset] = le.fit_transform(features[:, 1+offset])

    # encode job name
    le = preprocessing.LabelEncoder()
    features[:, 2+offset] = le.fit_transform(features[:, 2+offset])
    print('Preprocessing complete\n')

    return features, labels


def get_disk_data(interval='d', production=None):
    '''
    Read the Backblaze disk dataset from csv file
    The input path and filename are specified in macros
    Return features and labels after proper preprocessing
    
    Args:
        interval (chr): the interval of timestamp, by default day of year (d)
        Possible selections are day in a year (d) and month in a year (m)

    Returns:
        features (np.array): feature vector, the first column is the timestamp
        labels (np.array): True or False, binary classification
    '''
    path = os.path.join(INPUT_FOLDER, BACKBLAZE_INPUT_FILE)
    print('Loading data from', path)
    df = pd.read_csv(path, header=None)

    columns = ['serial_number', 'date',
        'smart_1_raw', 'smart_4_raw', 'smart_5_raw', 'smart_7_raw', 'smart_9_raw', 'smart_12_raw', 'smart_187_raw', 'smart_193_raw', 'smart_194_raw', 'smart_197_raw', 'smart_199_raw',
        'smart_4_raw_diff', 'smart_5_raw_diff', 'smart_9_raw_diff', 'smart_12_raw_diff', 'smart_187_raw_diff', 'smart_193_raw_diff', 'smart_197_raw_diff', 'smart_199_raw_diff',
        'label']
    df.columns = columns
    print('Load complete')

    #if production:
    #    if production not in ['Seagate', 'Western Digital', 'Hitachi']:
    #        print('Invalid production factory!')
    #        return
    #    print('Only choose disk from', production)
    #    production_df = pd.read_csv(os.path.join(INPUT_FOLDER, PRODUCTION_FILE))
    #    production_df = production_df[production_df['production'] == production]
    #    serial_set = set(production_df['serial_number'].to_list())
    #    df = df[df['serial_number'].isin(serial_set)]

    print('Preprocessing features')
    df = df[df.columns[1:]] # remove serial number
    # change the date into days of a year as all data are in 2015
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    if interval == 'd':
        df['date'] = pd.Series(pd.DatetimeIndex(df['date']).dayofyear)
    elif interval == 'm':
        df['date'] = pd.Series(pd.DatetimeIndex(df['date']).month)
    else: 
        print('Invalid time interval argument for reading disk failure data. Possible options are (d, m).')
        exit(-1)
    
    features = df[df.columns[:-1]].to_numpy()
    labels = df[df.columns[-1]].to_numpy()

    return features, labels


def obtain_metrics(labels, probas):
    '''
    Calculate performance on various metrics

    Args: 
        labels (np.array): labels of samples, should be True/False
        probas (np.array): predicted probabilities of samples, should be in [0, 1]
            and should be generated with predict_proba()[:, 1]
    Returns:
        (list): [ Precision, Recall, Accuracy, F-Measure, AUC, MCC, Brier Score ]
    '''
    preds = probas > 0.5
    ret = []
    ret.append(metrics.precision_score(labels, preds))
    ret.append(metrics.recall_score(labels, preds))
    ret.append(metrics.accuracy_score(labels, preds))
    ret.append(metrics.f1_score(labels, preds))
    ret.append(metrics.roc_auc_score(labels, probas))
    ret.append(metrics.matthews_corrcoef(labels, preds))
    ret.append(metrics.brier_score_loss(labels, probas))

    return ret
    

def obtain_model(model_name):
    '''
    This function instantiate a specific model 
    Note: the MODEL_TYPE global variable must be set first
    Args:
        model_name (str): [rf, nn, svm, cart, rgf]
    Returns:
        (instance): instance of given model with preset parameters.
        Return None if the model name is not in the option
    '''
    if model_name == 'rf':
        return RandomForestClassifier(n_estimators=50, criterion='gini', class_weight=None, max_depth=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2, n_jobs=N_WORKERS)
        #return RandomForestClassifier(n_jobs=N_WORKERS)
    elif model_name == 'nn':
        return MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, learning_rate='adaptive')
        #return MLPClassifier()
    elif model_name == 'svm':
        return SVC(max_iter=100000, probability=True)
        #return SVC(max_iter=10000, probability=True)
    elif model_name == 'cart':
        return DecisionTreeClassifier(criterion='gini', class_weight=None, max_depth=None, min_impurity_decrease=0.0, min_samples_leaf=1, min_samples_split=2)
        #return DecisionTreeClassifier()
    elif model_name == 'rgf':
        return SafeRGF()

    return None


def obtain_model_tuned(model_name, dataset):
    if dataset == 'g':
        if model_name == 'rf':
            return RandomForestClassifier(n_estimators=165, criterion='gini', bootstrap=True, class_weight='balanced', 
                                          max_depth=40, max_features='auto', min_samples_leaf=4, min_samples_split=8, 
                                          n_jobs=N_WORKERS)
        elif model_name == 'nn':
            return MLPClassifier(hidden_layer_sizes=(32, 16), activation='logistic', max_iter=520, learning_rate='constant', solver='adam', alpha=0.000662924906457569)
        elif model_name == 'svm':
            return SVC(max_iter=29613, class_weight=None, C=4.135537054059932, kernel='linear', probability=True)
        elif model_name == 'cart':
            return DecisionTreeClassifier(criterion='entropy', class_weight=None, max_depth=10, splitter='best',
                                        min_samples_split=2, min_samples_leaf=4, max_features='auto')
        elif model_name == 'rgf':
            return None
    elif dataset == 'b':
        if model_name == 'rf':
            return RandomForestClassifier(n_estimators=160, criterion='gini', bootstrap=False, class_weight='balanced', 
                                          max_depth=10, max_features='sqrt', min_samples_leaf=4, min_samples_split=8, 
                                          n_jobs=N_WORKERS)
        elif model_name == 'nn':
            return MLPClassifier(hidden_layer_sizes=(16,), activation='logistic', max_iter=520, learning_rate='constant', solver='adam', alpha=0.00023931672486512135)
        elif model_name == 'svm':
            return SVC(max_iter=81725, class_weight='balanced', C=1.6308757488530967, kernel='sigmoid', gamma=0.000202229364912411, probability=True)
        elif model_name == 'cart':
            return DecisionTreeClassifier(criterion='gini', class_weight=None, max_depth=10, splitter='random',
                                        min_samples_split=4, min_samples_leaf=4, max_features='sqrt')
        elif model_name == 'rgf':
            return None

    return None


def obtain_model_raw(model_name):
    '''
    This function instantiate a specific model 
    Note: the MODEL_TYPE global variable must be set first
    Args:
        model_name (str): [rf, nn, svm, cart, rgf]
    Returns:
        (instance): instance of given model with preset parameters.
        Return None if the model name is not in the option
    '''
    if model_name == 'rf':
        return RandomForestClassifier(n_jobs=N_WORKERS)
    elif model_name == 'nn':
        return MLPClassifier()
    elif model_name == 'svm':
        return SVC()
    elif model_name == 'cart':
        return DecisionTreeClassifier()
    elif model_name == 'rgf':
        return RGFClassifier()

    return None


def downsampling(training_features, training_labels, ratio=10):
    #return training_features, training_labels

    idx_true = np.where(training_labels == True)[0]
    idx_false = np.where(training_labels == False)[0]
    #print('Before dowmsampling:', len(idx_true), len(idx_false))
    idx_false_resampled = resample(idx_false, n_samples=len(idx_true)*ratio, replace=False)
    idx_resampled = np.concatenate([idx_false_resampled, idx_true])
    idx_resampled.sort()
    resampled_features = training_features[idx_resampled]
    resampled_labels = training_labels[idx_resampled]
    #print('After dowmsampling:', len(idx_true), len(idx_false_resampled))
    return resampled_features, resampled_labels


def time_based_splitting(features, labels, ratio):
    '''
    Split the data according to their timestamp
    Note that it assumes the first column is the timestamp
    '''
    count = int(np.round(len(features) * ratio))
    sort = features[:, 0].astype(np.int64).argsort()
    first_indexes = sort[:count]
    second_indexes = sort[count:]
    training_features = features[first_indexes]
    testing_features = features[second_indexes]
    training_labels = labels[first_indexes]
    testing_labels = labels[second_indexes]

    return training_features, testing_features, training_labels, testing_labels


def decay_function(n, tau, N0=1.0):
    '''
    Give the decay coeficients for given periods and parameters

    Args:
        n (int): number of periods
        tau (int): mean life time, coef=1/e when t=tau
        N0 (float): initial quantity, default by 1.0
    Returns:
        (np.array): 
    '''
    t = np.arange(n, 0, -1) - 1
    decay_coef = N0 * np.exp(-t / tau)
    return decay_coef
    
    
def obtain_intervals(dataset):
    '''
    Generate interval terminals, so that samples in each interval have:
        interval_i = (timestamp >= terminal_i) and (timestamp < terminal_{i+1})

    Args:
        dataset (chr): Assuming only Backblaze (b) and Google (g) datasets exists
    '''
    if dataset == 'g':
        # time unit in Google: millisecond, tracing time: 29 days
        start_time = 604046279
        unit_period = 24 * 60 * 60 * 1000 * 1000  # unit period: one day
        end_time = start_time + 28*unit_period
    elif dataset == 'b':
        # time unit in Backblaze: month, tracing time: one year (12 months)
        start_time = 1
        unit_period = 1  # unit period: one month
        end_time = start_time + 12*unit_period

    # add one unit for the open-end of range function
    terminals = [i for i in range(start_time, end_time+unit_period, unit_period)]

    return terminals
    

def obtain_natural_chunks(features, labels, terminals):
    feature_list = []
    label_list = []
    for i in range(len(terminals) - 1):
        idx = np.logical_and(features[:, 0] >= terminals[i], features[:, 0] < terminals[i + 1])
        feature_list.append(features[idx][:, 1:])
        label_list.append(labels[idx])
    return feature_list, label_list


def obtain_chunks(features, labels, N):
    '''
    Split data into N consecutive chunks
    If the size of the last chunk is smaller, it will be merged into the second last chunk
    Return a list of chunks for features and labels
    '''
    feature_list = []
    label_list = []
    n_samples = features.shape[0] // N
    for i in range(N):
        if i != N - 1:
            feature_list.append(features[i*n_samples: (i + 1)*n_samples])
            label_list.append(labels[i*n_samples: (i + 1)*n_samples])
        else:
            feature_list.append(features[i*n_samples:])
            label_list.append(labels[i*n_samples:])
            break

    print([len(label) for label in label_list])
    return feature_list, label_list


class SafeRGF(RGFClassifier):
    def __init__(self,
                 max_leaf=1000,
                 test_interval=100,
                 algorithm="RGF_Sib",
                 loss="Log",
                 reg_depth=1.0,
                 l2=0.1,
                 sl2=None,
                 normalize=False,
                 min_samples_leaf=10,
                 n_iter=None,
                 n_tree_search=1,
                 opt_interval=100,
                 learning_rate=0.5,
                 calc_prob="sigmoid",
                 n_jobs=1,
                 memory_policy="generous",
                 verbose=0,
                 init_model=None):
        super(SafeRGF, self).__init__()
        #self.max_leaf = max_leaf
        #self.test_interval = test_interval
        #self.algorithm = algorithm
        #self.loss = loss
        #self.reg_depth = reg_depth
        #self.l2 = l2
        #self.sl2 = sl2
        #self.normalize = normalize
        #self.min_samples_leaf = min_samples_leaf
        #self.n_iter = n_iter
        #self.n_tree_search = n_tree_search
        #self.opt_interval = opt_interval
        #self.learning_rate = learning_rate
        #self.calc_prob = calc_prob
        self.n_jobs = n_jobs
        #self.memory_policy = memory_policy
        #self.verbose = verbose
        #self.init_model = init_model
        self.is_foul = False

    def fit(self, X, y):
        try:
            self.is_foul = False
            super(SafeRGF, self).fit(X, y)
        except Exception:
            self.is_foul = True
            traceback.print_exc()
            print('Shape of features:', X.shape)
            print('Ratios of labels:', np.count_nonzero(y), '/', np.count_nonzero(y==0))
        
    def predict_proba(self, X):
        if self.is_foul:
            return np.hstack((np.ones((X.shape[0], 1)), np.zeros((X.shape[0], 1))))
        else:
            return super(SafeRGF, self).predict_proba(X)
    
    def predict(self, X):
        if self.is_foul:
            return np.zeros(X.shape[0]).astype(bool)
        else:
            return super(SafeRGF, self).predict(X)

