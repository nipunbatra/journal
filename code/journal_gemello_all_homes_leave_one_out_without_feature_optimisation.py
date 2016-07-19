"""
This code generates the prediction for a region when we use homes containing all data


"""

# NEED TO RUN ON CLUSTER
import sys

CLUSTER = True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import pickle
from collections import OrderedDict

out_overall = pickle.load(open('../data/input/all_regions.pkl', 'r'))

region = "Austin"

df = out_overall[region]
df = df.rename(columns={'house_num_rooms': 'num_rooms',
                                  'num_occupants': 'total_occupants',
                                  'difference_ratio_min_max': 'ratio_difference_min_max'})

df = df[(df.full_agg_available == 1) & (df.md_available == 1)]


def scale_0_1(ser, minimum=None, maximum=None):
    if minimum is not None:
        pass
    else:
        minimum = ser.min()
        maximum = ser.max()
    return (ser - minimum).div(maximum - minimum)

def normalise(df):
    new_df = df.copy()
    max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
    min_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].min().min()
    new_df[["aggregate_%d" % i for i in range(1, 13)]] = scale_0_1(df[["aggregate_%d" % i for i in range(1, 13)]],
                                                                   min_aggregate, max_aggregate)
    for col in ['area', 'total_occupants', 'num_rooms', 'ratio_min_max',
                'skew', 'kurtosis', 'variance', 'ratio_difference_min_max', 'p_25',
                'p_50', 'p_75']:
        new_df[col] = scale_0_1(df[col])
    return new_df

df = normalise(df)

from all_functions import *
from features import *

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit

NUM_NEIGHBOUR_MAX = 6
F_MAX = 6

K_min, K_max = 1,6
F_min, F_max=1,8

import json

from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import KFold
from sklearn.ensemble import ExtraTreesRegressor

def _save_csv(out_df, path, appliance, num_homes, start_seed, end_seed, feature):
    out_df.T.to_csv("%s/%s_%d_%d_%d_%s.csv" %(path, appliance, num_homes, start_seed, end_seed, feature),
                    index_label="Random seed")


def _find_accuracy(home, appliance, feature="Monthly"):
    np.random.seed(42)
    appliance_df = df.ix[all_homes]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    test_homes = [home]
    train_homes = appliance_df[~appliance_df.index.isin([home])].index


    #all_home_appliance = deepcopy(all_homes)
    #all_home_appliance[appliance] = train_homes

    # Cross validation on inner loop to find best feature, K
    train_size = len(train_homes)
    l = LeaveOneOut(train_size)
    out = OrderedDict()
    error_K ={}
    pred_K ={}
    for K in range(K_min, K_max):
        for cv_train, cv_test in l:

            cv_train_home=appliance_df.ix[train_homes[cv_train]]
            cv_test_home = appliance_df.ix[train_homes[cv_test]]
            test_home_name = cv_test_home.index.values[0]
            #print cv_test_home
            out[test_home_name]={}

            # Now fitting KNN on this
            for month in range(start, stop):

                clf = KNeighborsRegressor(n_neighbors=K)
                clf.fit(cv_train_home[feature_map[feature]], cv_train_home['%s_%d' %(appliance, month)])
                out[test_home_name][month] =clf.predict(cv_test_home[feature_map[feature]])[0]
        pred =pd.DataFrame(out).T
        #print pred
        pred_K[K] =pred

        gt = appliance_df[['%s_%d' %(appliance, i) for i in range(start, stop)]].copy().drop(test_home_name)
        #print gt
        pred.columns = gt.columns
        error = (pred-gt).abs().div(gt).mul(100)
        error_K[K] = error.median().mean()

    error_K=pd.Series(error_K)

    K_best=error_K.argmin()
    # Now predicting for test home
    train_overall = appliance_df.ix[appliance_df[~appliance_df.index.isin([home])].index]
    test_overall = appliance_df[appliance_df.index.isin([home])]
    pred_test = {}
    gt_test = {}
    for month in range(start, stop):
        clf = KNeighborsRegressor(n_neighbors=K_best)
        clf.fit(train_overall[feature_map[feature]], train_overall['%s_%d' %(appliance, month)])
        pred_test[month] = clf.predict(test_overall[feature_map[feature]])
        gt_test[month] = test_overall['%s_%d' %(appliance, month)]


    #json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open("../main-out-new/%s_%s_%d.json" %(appliance,feature, home),"w") )

    pred_df = pd.DataFrame(pred_test)
    pred_df.index = [home]
    gt_df = pd.DataFrame(gt_test)
    error = (gt_df-pred_df).abs().div(gt_df).mul(100)
    F_best = feature_map[feature]
    return pred_df, gt_df, error, F_best, K_best




import os

out_path = os.path.expanduser("~/output/journal/gemello/all_homes_without_feature_optimisation/")
import sys
appliance, feature, home = sys.argv[1], sys.argv[2], sys.argv[3]
home = int(home)

if appliance=="hvac":
    start, stop=5, 11
else:
    start, stop=1, 13

appliance_df = df.ix[df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
all_homes = appliance_df.index

pred_df, gt_df, error, F_best, K_best = _find_accuracy(home, appliance, feature)


if not os.path.exists(out_path):
    print "here"
    os.makedirs(out_path)
import pickle
filename = os.path.join(out_path, "%s_%s_%d.pkl" %(appliance,feature,home))
o = {'pred_df':pred_df,'gt_df':gt_df,'error':error,
     'F_best':F_best,'K_best':K_best}
pickle.dump(o, open(filename,'w'))
#_save_csv(out_overall, os.path.expanduser("~/output/unified/kdd_all_features/"), appliance, num_homes, start_seed, end_seed, feature)
