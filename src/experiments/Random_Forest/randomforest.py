import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from lifelines.utils import concordance_index
from sklearn.metrics import mean_squared_error
from ast import literal_eval
import json

def r_squared_error(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean))

    return mult / float(y_obs_sq * y_pred_sq)

def squared_error_zero(y_obs, y_pred):
    k = get_k(y_obs, y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k * y_pred)))
    down= sum((y_obs - y_obs_mean) * (y_obs - y_obs_mean))

    return 1 - (upp / float(down))

def get_k(y_obs, y_pred):
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))
def get_rm2(ys_orig, ys_line):
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)
    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))


if __name__ == '__main__':
    # Setting
    is_ssh = True
    # Setting = 1 -> ecfp+uni
    # Setting = 2 -> cddd+uni
    # Setting = 3 -> ecfp+seq2vec
    setting = 3


    ssh_path = ''
    # Affinities
    path_train_aff_KIBA = ssh_path + 'data/affi/train_aff_KIBA.pkl'
    path_test_aff_KIBA = ssh_path + 'data/affi/test_aff_KIBA.pkl'
    # ECFP
    path_train_ecfp = ssh_path + 'descriptors/ecfp_train.npy'
    path_test_ecfp = ssh_path + 'descriptors/ecfp_test.npy'
    # CDDD
    path_train_cddd = ssh_path + 'descriptors/cddd_train.txt'
    path_test_cddd = ssh_path + 'descriptors/cddd_test.txt'
    # UniRep
    path_train_uni = ssh_path + 'descriptors/uni_train.npy'
    path_test_uni = ssh_path + 'descriptors/uni_test.npy'
    # Seq2Vec
    path_train_seq = ssh_path + 'descriptors/SeqVec_train.npy'
    path_test_seq = ssh_path + 'descriptors/SeqVec_test.npy'

    #get labels
    # train
    train_affi_KIBA = pd.read_pickle(path_train_aff_KIBA)
    y_train_KIBA = train_affi_KIBA['affinity'].tolist()
    # test
    test_affi_KIBA = pd.read_pickle(path_test_aff_KIBA)
    y_test_KIBA = test_affi_KIBA['affinity'].tolist()

    if setting == 1:
        # setting1: ecfp+uni train
        #load features
        train_ecfp = np.load(path_train_ecfp, allow_pickle=True).tolist()
        test_ecfp = np.load(path_test_ecfp, allow_pickle=True).tolist()
        train_uni = np.load(path_train_uni, allow_pickle=True).tolist()
        test_uni = np.load(path_test_uni, allow_pickle=True).tolist()


        print(f'ecfp train: {len(train_ecfp)}\t train[0]: {len(train_ecfp[0])}')
        print(f'ecfp test: {len(test_ecfp)}\t test[0]: {len(test_ecfp[0])}')
        print(f'uni train: {len(train_uni)}\t train[0]: {len(train_uni[0])}')
        print(f'uni test: {len(test_uni)}\t test[0]: {len(test_uni[0])}')

        train_ecfp_uni = []
        for idx, ecfp in enumerate(train_ecfp):
            train_ecfp_uni.append(ecfp+train_uni[idx])
        test_ecfp_uni = []
        for idx, ecfp in enumerate(test_ecfp):
            test_ecfp_uni.append(ecfp+test_uni[idx])
        print(f'ecfp+uni train: {len(train_ecfp_uni)}\t train[0]: {len(train_ecfp_uni[0])}')
        print(f'ecfp+uni test: {len(test_ecfp_uni)}\t test[0]: {len(test_ecfp_uni[0])}')


        #fit regressor ecfp + uni
        classifier_ecfp_uni = RandomForestRegressor(n_estimators=1001)
        classifier_ecfp_uni.fit(train_ecfp_uni, y_train_KIBA)

        pred_ecfp_uni= classifier_ecfp_uni.predict(test_ecfp_uni)
        results_ecfp_uni = {
            'ConcordanceIndex' : concordance_index(y_test_KIBA, pred_ecfp_uni),
            'MSE' : mean_squared_error(y_test_KIBA, pred_ecfp_uni),
            'rm2' : get_rm2(y_test_KIBA, pred_ecfp_uni)
        }
        print("ecfp+uni")
        print(f'Concordance index:\t {results_ecfp_uni["ConcordanceIndex"]}')
        print(f'MSE\t\t：{results_ecfp_uni["MSE"]}')
        print(f'rm2\t\t：{results_ecfp_uni["rm2"]}')

        with open('results_ecfp_uni.json', 'w') as outfile:
            json.dump(results_ecfp_uni, outfile)
    elif setting == 2:
        train_uni = np.load(path_train_uni, allow_pickle=True).tolist()
        test_uni = np.load(path_test_uni, allow_pickle=True).tolist()

        train_cddd = []
        with open(path_train_cddd, 'r') as fp:
            for line in fp:
                x = line[:-1]
                if x == 'nan':
                    train_cddd.append(x)
                else:
                    train_cddd.append(literal_eval(x))

        test_cddd = []
        with open(path_test_cddd, 'r') as fp:
            for line in fp:
                x = line[:-1]
                if x == 'nan':
                    test_cddd.append(x)
                else:
                    test_cddd.append(literal_eval(x))

        print(f'cddd train: {len(train_cddd)}\t train[1]: {len(train_cddd[1])}')  # first cddd is nan
        print(f'cddd test: {len(test_cddd)}\t test[0]: {len(test_cddd[0])}')
        print(f'uni train: {len(train_uni)}\t train[0]: {len(train_uni[0])}')
        print(f'uni test: {len(test_uni)}\t test[0]: {len(test_uni[0])}')

        # setting2: cddd+uni train
        # clean: nan have to be removed from cddd
        # new label are needed!
        train_cddd_uni = []
        y_train_KIBA_cleaned = []
        for idx, cddd in enumerate(train_cddd):
            if str(cddd) != 'nan':
                train_cddd_uni.append(cddd + train_uni[idx])
                y_train_KIBA_cleaned.append(y_train_KIBA[idx])

        test_cddd_uni = []
        y_test_KIBA_cleaned = []
        for idx, cddd in enumerate(test_cddd):
            if str(cddd) != 'nan':
                test_cddd_uni.append(cddd + test_uni[idx])
                y_test_KIBA_cleaned.append(y_test_KIBA[idx])

        print(f'cddd+uni train: {len(train_cddd_uni)}\t train[0]: {len(train_cddd_uni[0])}')
        print(f'cddd+uni train label: {len(y_train_KIBA_cleaned)}')
        print(f'cddd+uni test: {len(test_cddd_uni)}\t test[0]: {len(test_cddd_uni[0])}')
        print(f'cddd+uni test label: {len(y_test_KIBA_cleaned)}')

        # fit regressor cddd + uni
        classifier_cddd_uni = RandomForestRegressor(n_estimators=1001)
        classifier_cddd_uni.fit(train_cddd_uni, y_train_KIBA_cleaned)

        pred_cddd_uni = classifier_cddd_uni.predict(test_cddd_uni)
        results_cddd_uni = {
            'ConcordanceIndex': concordance_index(y_test_KIBA_cleaned, pred_cddd_uni),
            'MSE': mean_squared_error(y_test_KIBA_cleaned, pred_cddd_uni),
            'rm2': get_rm2(y_test_KIBA_cleaned, pred_cddd_uni)
        }
        print("cddd+uni")
        print(f'Concordance index:\t {results_cddd_uni["ConcordanceIndex"]}')
        print(f'MSE\t\t：{results_cddd_uni["MSE"]}')
        print(f'rm2\t\t：{results_cddd_uni["rm2"]}')

        with open('results_cddd_uni.json', 'w') as outfile:
            json.dump(results_cddd_uni, outfile)

    elif setting == 3:
        # setting2: cddd+uni train
        # load features
        train_ecfp = np.load(path_train_ecfp, allow_pickle=True).tolist()
        test_ecfp = np.load(path_test_ecfp, allow_pickle=True).tolist()
        train_seq = np.load(path_train_seq, allow_pickle=True).tolist()
        test_seq = np.load(path_test_seq, allow_pickle=True).tolist()

        print(f'ecfp train: {len(train_ecfp)}\t train[0]: {len(train_ecfp[0])}')
        print(f'ecfp test: {len(test_ecfp)}\t test[0]: {len(test_ecfp[0])}')
        print(f'seq train: {len(train_seq)}\t train[0]: {len(train_seq[0])}')
        print(f'seq test: {len(test_seq)}\t test[0]: {len(test_seq[0])}')

        train_ecfp_seq = []
        for idx, ecfp in enumerate(train_ecfp):
            train_ecfp_seq.append(ecfp + train_seq[idx])
        test_ecfp_seq = []
        for idx, ecfp in enumerate(test_ecfp):
            test_ecfp_seq.append(ecfp + test_seq[idx])
        print(f'ecfp+seq train: {len(train_ecfp_seq)}\t train[0]: {len(train_ecfp_seq[0])}')
        print(f'ecfp+seq test: {len(test_ecfp_seq)}\t test[0]: {len(test_ecfp_seq[0])}')

        # fit regressor ecfp + uni
        classifier_ecfp_seq = RandomForestRegressor(n_estimators=1001)
        classifier_ecfp_seq.fit(train_ecfp_seq, y_train_KIBA)

        pred_ecfp_seq = classifier_ecfp_seq.predict(test_ecfp_seq)
        results_ecfp_seq = {
            'ConcordanceIndex': concordance_index(y_test_KIBA, pred_ecfp_seq),
            'MSE': mean_squared_error(y_test_KIBA, pred_ecfp_seq),
            'rm2': get_rm2(y_test_KIBA, pred_ecfp_seq)
        }
        print("ecfp+seq")
        print(f'Concordance index:\t {results_ecfp_seq["ConcordanceIndex"]}')
        print(f'MSE\t\t：{results_ecfp_seq["MSE"]}')
        print(f'rm2\t\t：{results_ecfp_seq["rm2"]}')

        with open('results_ecfp_seq.json', 'w') as outfile:
            json.dump(results_ecfp_seq, outfile)