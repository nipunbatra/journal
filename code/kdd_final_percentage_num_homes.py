# NEED TO RUN ON CLUSTER
import sys

CLUSTER = True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import numpy as np
import pandas as pd
import pickle

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

df_unnormalised = df.copy()
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


def _find_accuracy_num_homes(appliance, num_homes, start_seed, end_seed, feature="Monthly"):
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13
    out = {}
    out_overall={}
    # We need to find homes that have all the features
    appliance_df = df.ix[df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
    all_homes = appliance_df.index
    kf = KFold(len(all_homes), n_folds=5)
    for cv_loop_index, (train_index, test_index) in enumerate(kf):
        out_overall[cv_loop_index] = {}
        train_df = appliance_df.ix[all_homes[train_index]]
        test_df = appliance_df.ix[all_homes[test_index]]
        #print train_df.index
        print "TRAINING>>>"

        #Now, for each random seed, we'll pick up `num_homes` homes from the train set
        # Do CV on that to pick up best features and then predict for the test homes
        error_df_list = {}
        #1. Multiple times we will choose `num_homes` from the train set
        for random_seed in range(start_seed, end_seed):
            print "Random seed:", random_seed



            #out_overall[random_seed] = {}
            train_subset_homes_idx = np.random.choice(len(train_df), num_homes, replace=False)
            train_subset_homes = train_df.ix[train_df.index[train_subset_homes_idx]]
            #print train_subset_homes
            #2. Now, on this small subset of homes, we will do a round of CV to learn optimum
            # features

            l = LeaveOneOut(len(train_subset_homes_idx))
            for cv_train, cv_test in l:
                cv_train_home =appliance_df.ix[train_subset_homes.index[cv_train]]
                cv_test_home = appliance_df.ix[train_subset_homes.index[cv_test]]
                test_home_name = cv_test_home.index.values[0]
                Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
                forest = ExtraTreesRegressor(n_estimators=250,
                                      random_state=0)
                forest.fit(cv_train_home[feature_map[feature]], Y)
                importances = forest.feature_importances_
                indices = np.argsort(importances)[::-1]

                # Now varying K and top-N features
                out[test_home_name] ={}
                for K in range(K_min, K_max):
                    out[test_home_name][K]={}
                    for top_n in range(F_min,F_max):
                        out[test_home_name][K][top_n]=[]
                        top_n_features = cv_train_home[feature_map[feature]].columns[indices][:top_n]

                        # Now fitting KNN on this
                        for month in range(start, stop):
                            clf = KNeighborsRegressor(n_neighbors=K)
                            clf.fit(cv_train_home[top_n_features], cv_train_home['%s_%d' %(appliance, month)])
                            dist, ind = clf.kneighbors(cv_test_home[top_n_features])
                            nghbrs = cv_train_home.index.values[ind].flatten()
                            proportion = cv_train_home.ix[nghbrs]['%s_%d' %(appliance, month)].div(df_unnormalised.ix[nghbrs]['%s_%d' %("aggregate", month)])
                            mean_prop = proportion.mean()
                            out[test_home_name][K][top_n].append(df_unnormalised.ix[cv_test_home.index]['%s_%d' %("aggregate", month)]*mean_prop)

            accur = {}

            # We want to find the F, K combination that minimised the median (over homes)
            # and mean over months error
            for K in range(K_min, K_max):
                accur[K] = {}
                for top_n in range(F_min, F_max):
                    accur[K][top_n]={}
                    temp = {}
                    for h in out.iterkeys():
                        pred = pd.DataFrame(out[h][K][top_n]).T
                        pred.index = [h]
                        pred.columns = [['%s_%d' %(appliance, i) for i in range(start, stop)]]
                        gt = appliance_df.ix[h][['%s_%d' %(appliance, i) for i in range(start, stop)]]
                        error = (pred-gt).abs().div(gt).mul(100).squeeze()
                        accur[K][top_n][h]=error

                    accur[K][top_n] = pd.DataFrame(accur[K][top_n]).T.median().mean()

            accur_df = pd.DataFrame(accur)
            accur_min = accur_df.min().min()
            min_ac_df = accur_df[accur_df==accur_min]
            F_best = cv_train_home[feature_map[feature]].columns[indices][:min_ac_df.mean(axis=1).dropna().index.values[0]].tolist()
            K_best = min_ac_df.mean().dropna().index.values[0]

            # Now predicting for test home


            pred_test = {}
            gt_test = {}
            for month in range(start, stop):
                clf = KNeighborsRegressor(n_neighbors=K_best)
                clf.fit(train_subset_homes[F_best], train_subset_homes['%s_%d' %(appliance, month)])
                dist, ind = clf.kneighbors(test_df[F_best])
                nghbrs = train_subset_homes[F_best].index.values[ind].flatten()[:K]
                nr = train_subset_homes.ix[nghbrs]['%s_%d' %(appliance, month)]
                dr = df_unnormalised.ix[nghbrs]['%s_%d' %("aggregate", month)]
                nr.name = dr.name
                proportion =nr.div(dr)
                mean_prop = proportion.mean()
                pred_test[month] =df_unnormalised.ix[test_df.index]['%s_%d' %("aggregate", month)]*mean_prop
                gt_test[month] = test_df['%s_%d' %(appliance, month)]


            #json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open("../sensitivity-new/%s_%s_%d.json" %(appliance,feature, home),"w") )

            pred_df = pd.DataFrame(pred_test)
            pred_df.index = test_df.index
            gt_df = pd.DataFrame(gt_test)
            error = (gt_df-pred_df).abs().div(gt_df).mul(100)
            out_overall[cv_loop_index][random_seed] = error

    errors = {}
    for random_seed in range(start_seed, end_seed):
        temp_list = []
        for cv_loop_index in range(len(kf)):
            temp_list.append(out_overall[cv_loop_index][random_seed])
        errors[random_seed] = pd.concat(temp_list)


    return errors


import time
import sys, os
appliance="hvac"
start_seed=0
end_seed=25
feature="Monthly+Static"
num_homes = int(sys.argv[1])
print
a= time.time()
print "*"*40
print num_homes
print "*"*40
num_homes = int(num_homes)
if num_homes<K_max+1:
    K_max=num_homes-1
start_seed=int(start_seed)
end_seed=int(end_seed)
o= _find_accuracy_num_homes(appliance, num_homes, start_seed, end_seed, feature)
out_path = os.path.expanduser("~/output/unified/kdd_all_features_cv_percentage/")
if not os.path.exists(out_path):
    print "here"
    os.makedirs(out_path)
import pickle
filename = os.path.join(out_path, "%s_%s_%d_%d_%d_%s.pkl" %(region,appliance, num_homes, start_seed, end_seed, feature))
pickle.dump(o, open(filename,'w'))
b = time.time()
print "Took", b-a, "seconds for ", num_homes
#_save_csv(out_overall, os.path.expanduser("~/output/unified/kdd_all_features/"), appliance, num_homes, start_seed, end_seed, feature)