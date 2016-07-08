# NEED TO RUN ON CLUSTER
import sys
CLUSTER=True
if CLUSTER:
    sys.path.insert(0, '/if6/nb2cz/anaconda/lib/python2.7/site-packages')

import  os

import numpy as np
import pandas as pd


from code.create_df_larger import read_df_larger
df, dfc, all_homes, appliance_min, national_average = read_df_larger()

df = df.rename(columns={'house_num_rooms':'num_rooms',
                        'num_occupants':'total_occupants',
                        'difference_ratio_min_max':'ratio_difference_min_max'})
K_min, K_max = 1,6
F_min, F_max=1,8

from code.all_functions import *
from code.features_larger import *

import sys

from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_validation import ShuffleSplit
from sklearn.cross_validation import LeaveOneOut

NUM_NEIGHBOUR_MAX = 6
F_MAX = 6

import json



from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from collections import OrderedDict

def _find_accuracy(home, appliance, feature="Monthly", num_homes=5):
    np.random.seed(42)
    appliance_df = df.ix[all_homes[appliance]]
    if appliance=="hvac":
        start, stop=5, 11
    else:
        start, stop=1, 13

    test_homes = [home]
    train_d = appliance_df[~appliance_df.index.isin([home])]
    train_d_index = train_d[['%s_%d' %(appliance, i) for i in range(start, stop)]].dropna().index
    train_d_feature = train_d.ix[train_d_index][feature_map[feature]].dropna()

    from sklearn.cluster import KMeans
    c = KMeans(n_clusters=num_homes)
    c.fit(train_d_feature)
    to_use = []
    for i in range(num_homes):
        d = c.transform(train_d_feature)[:, i]
        ind = np.argsort(d)[::-1][:num_homes]
        flag=False
        start_index = 0
        while flag is False:

            if train_d_feature.index.values[ind[start_index]] not in to_use:
                to_use.append(train_d_feature.index.values[ind[start_index]])
                flag=True
            else:
                start_index = start_index+1

    train_homes = np.array(to_use)
    all_home_appliance = deepcopy(all_homes)
    all_home_appliance[appliance] = train_homes

    # Cross validation on inner loop to find best feature, K
    train_size = len(train_homes)
    l = LeaveOneOut(train_size)
    out = OrderedDict()
    for cv_train, cv_test in l:

        cv_train_home=appliance_df.ix[train_homes[cv_train]]
        cv_train_index = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].dropna().index
        cv_train_home = cv_train_home.ix[cv_train_index]
        cv_test_home = appliance_df.ix[train_homes[cv_test]]
        test_home_name = cv_test_home.index.values[0]
        #print cv_test_home
        out[test_home_name]={}


        # Summing up energy across start to stop to get Y to learn optimum feature on
        Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
        forest = ExtraTreesRegressor(n_estimators=250,
                              random_state=0)
        forest.fit(cv_train_home[feature_map[feature]], Y)
        importances = forest.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Now varying K and top-N features

        for K in range(K_min, K_max):
            out[test_home_name][K]={}
            for top_n in range(F_min,F_max):
                out[test_home_name][K][top_n]=[]
                top_n_features = cv_train_home[feature_map[feature]].columns[indices][:top_n]

                # Now fitting KNN on this
                for month in range(start, stop):
                    clf = KNeighborsRegressor(n_neighbors=K)
                    clf.fit(cv_train_home[top_n_features], cv_train_home['%s_%d' %(appliance, month)])
                    #print clf.predict(cv_test_home[top_n_features]), month
                    out[test_home_name][K][top_n].append(clf.predict(cv_test_home[top_n_features]))

        # Now, finding the (K, top_n) combination that gave us best accuracy on CV test homes
    accur = {}

    for K in range(K_min, K_max):
        accur[K] = {}
        for top_n in range(F_min, F_max):
            temp = {}
            for h in out.iterkeys():
                pred = pd.DataFrame(out[h][K][top_n]).T
                #all_but_h = [x for x in out.keys() if x!=h]
                pred.index = [h]
                pred.columns = [['%s_%d' %(appliance, i) for i in range(start, stop)]]
                gt = appliance_df.ix[h][['%s_%d' %(appliance, i) for i in range(start, stop)]]
                error = (pred-gt).abs().div(gt).mul(100)
                mean_error = error.mean().mean()
                a = 100-mean_error
                if a<0:
                    a=0
                temp[h]=a
            #ac = pd.Series(temp).mean()
            ac = pd.Series(temp).median()

            accur[K][top_n] = ac

    accur_df = pd.DataFrame(accur)
    accur_max = accur_df.max().max()
    max_ac_df = accur_df[accur_df==accur_max]
    F_best = cv_train_home[feature_map[feature]].columns[indices][:max_ac_df.mean(axis=1).dropna().index.values[0]].tolist()
    K_best = max_ac_df.mean().dropna().index.values[0]

    # Now predicting for test home
    train_overall = appliance_df.ix[appliance_df[~appliance_df.index.isin([home])].index]
    test_overall = appliance_df[appliance_df.index.isin([home])]
    pred_test = {}
    gt_test = {}
    for month in range(start, stop):
        clf = KNeighborsRegressor(n_neighbors=K_best)
        clf.fit(train_overall[F_best], train_overall['%s_%d' %(appliance, month)])
        pred_test[month] = clf.predict(test_overall[F_best])
        gt_test[month] = test_overall['%s_%d' %(appliance, month)]


    json.dump({'f':F_best, 'k':K_best,'accuracy':accur_max},open(os.path.expanduser("~/main-out-new-larger-num-homes/%d_%s_%s_%d.json" %(num_homes, appliance,feature, home)),"w") )

    pred_df = pd.DataFrame(pred_test)
    pred_mean = df.ix[train_homes][['%s_%d' %(appliance, month) for month in range(start, stop)]].mean()
    pred_df.index = [home]
    #gt_df = pd.DataFrame(gt_test)
    #print pred_df, gt_df
    #error = (gt_df-pred_df).abs().div(gt_df).mul(100)
    #print error
    #accuracy_test = 100-error
    #accuracy_test[accuracy_test<0]=0
    gt_df =df.ix[home][['%s_%d' %(appliance, month) for month in range(start, stop)]]
    gt_df.index = pred_df.columns
    pred_mean.index = pred_df.columns
    #return accuracy_test.squeeze()
    return pred_df, pred_mean,gt_df



import sys
appliance, feature, home, num_homes = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
home = int(home)
num_homes = int(num_homes)

pred_df, mean_df, gt_df = _find_accuracy(home, appliance, feature, num_homes)
pred_df.to_csv(os.path.expanduser("~/main-out-new-larger-num-homes-median/%d_%s_%s_%d.csv" %(num_homes, appliance,feature, home)))
mean_df.to_csv(os.path.expanduser("~/main-out-new-larger-num-homes-median-mean/%d_%s_%s_%d.csv" %(num_homes, appliance,feature, home)))
