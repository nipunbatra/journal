import pickle

import numpy as np
import pandas as pd

import sys
sys.path.append('../code/')

from degree_days import  dd
from regional_average_contribution import  contribution

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

import sys
import os

region, test_home, appliance, K = sys.argv[1:]
test_home = int(test_home)
K = int(K)

df = out_overall[region]
df = df[(df.full_agg_available == 1) & (df.md_available == 1)]


from itertools import combinations

start_month, end_month = 1,12
agg_features = np.hstack([['aggregate_'+str(month) for month in range(start_month, end_month+1)],
                         'ratio_min_max','difference_ratio_min_max','p_25','p_50','p_75','skew','kurtosis'])
md_features = ['area','house_num_rooms']
features = {'md_agg':np.hstack([
            agg_features,
            md_features
            ]).tolist()}

f_all = features['md_agg']

# Find not null set of common features
def find_com_features_train(df, home_1, home_2, featureset_max):
    f_1 = df.ix[home_1][featureset_max].dropna()
    f_2 = df.ix[home_2][featureset_max].dropna()
    com_f =  np.intersect1d(f_1.index, f_2.index)
    #print com_f
    return com_f

def find_distance_train_test(df_train, home_1, home_2, df_test, home_test, featureset_train, featureset_max):
    f_test = df_test[featureset_max].dropna()
    com_f =  np.intersect1d(f_test.index, featureset_train)
    if len(com_f):
        is_common = True
    else:
        is_common = False
        return is_common, None

    if len(com_f):
        a = np.linalg.norm(df_train.ix[home_1][com_f]- df_test[com_f])
        b = np.linalg.norm(df_train.ix[home_2][com_f]- df_test[com_f])
        if a<=b:
            order = [home_1, home_2]
        else:
            order = [home_2, home_1]
        return is_common, {'order':order,
                    'num_f':len(com_f),
                    'dist_a':a,
                    'dist_b':b,
                          'f':com_f}

import pandas as pd

def scale_0_1(ser, minimum=None, maximum=None):
    if minimum is not None:
        pass
    else:
        minimum = ser.min()
        maximum = ser.max()
    return (ser-minimum).div(maximum-minimum)

def normalise(df):
    new_df = df.copy()
    max_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].max().max()
    min_aggregate = df[["aggregate_%d" % i for i in range(1, 13)]].min().min()
    new_df[["aggregate_%d" % i for i in range(1, 13)]] = scale_0_1(df[["aggregate_%d" % i for i in range(1, 13)]], min_aggregate, max_aggregate)
    for col in ['area','num_occupants','house_num_rooms','ratio_min_max',
                'skew','kurtosis','variance','difference_ratio_min_max','p_25',
               'p_50','p_75']:
        new_df[col] = scale_0_1(df[col])
    return new_df


normalised_df = normalise(df)

train_df = df.copy().drop(test_home)
test_df = df.copy().ix[test_home]

train_normalised_df = normalised_df.copy().drop(test_home)
test_normalised_df = normalised_df.ix[test_home]


def solve_ilp(inequalities, time_limit=50):
    from collections import defaultdict
    import pandas as pd
    co = defaultdict(int)
    for ineq in inequalities:
        lt = ineq[0]
        gt = ineq[1]
        co[lt]-= 1
        co[gt]+= 1
    co_ser = pd.Series(co)
    co_ser.sort()

    return co_ser.index.values.tolist()

if appliance=="hvac":
    month_start, month_end = 5, 11
else:
    month_start, month_end=1,13
pred = {}
gt_data={}
error={}
for month_compute in range(month_start, month_end):

    num_features_all = {}
    ineq_dict = {}

    num_features_all[appliance] = {}
    ineq_dict[appliance] = {}

    #num_features_all[appliance][month_compute] = {}
    ineq_dict[appliance][month_compute] = {}

    candidate_homes = train_normalised_df['%s_%d' %(appliance, month_compute)].dropna().index.values
    # Removing the actual test home!
    candidate_homes = np.array(np.setdiff1d(candidate_homes, test_home))

    print len(candidate_homes), candidate_homes
    #num_features_all[appliance][month_compute][test_home] = defaultdict(int)
    from collections import defaultdict
    import pandas as pd
    co = defaultdict(int)
    store_path = '../../../output/journal/vistrit/all_features/'
    filename = '%s_%s_%d_%d.pkl' %(region,
                                                                        appliance,
                                                                        test_home, K)
    print store_path
    if os.path.exists(store_path):
        print "already exists"
        #continue
    else:
        os.makedirs(store_path)


    if not np.isnan(test_normalised_df['%s_%d' %(appliance, month_compute)]):
        # We need to predict this value!
        # Find candidate set, train homes which have not null for this month
        # Now find features on pairs of homes in candidate homes
        for a,b in combinations(candidate_homes, 2):
            com_features = find_com_features_train(train_normalised_df, a, b, f_all)

            if len(com_features)>15:
                # Consider a,b
                is_common, d = find_distance_train_test(train_normalised_df, a, b, test_normalised_df, test_home, com_features, f_all)
                if is_common:

                    # Common between train and test. Can add this pair to inequalities
                    ineq=d['order']
                    lt = ineq[0]
                    gt = ineq[1]
                    co[lt]-= 1
                    co[gt]+= 1
                    #num_features_all[appliance][month_compute][test_home][d['num_f']]+= 1

        """
        # Saving ineqs
        pickle.dump(ineqs, open('../data/model/inequalities/%s_%s_%s_%s_%d_%d.pkl' %(train_region,
                                                                        test_region,
                                                                        transform,
                                                                        appliance,
                                                                        month_compute,
                                                                        test_home),'w'))
        """
        co_ser = pd.Series(co)
        co_ser.sort()
        ranks = co_ser.index.values.tolist()

        pred[month_compute] = train_df.ix[ranks[:K]]['%s_%d' %(appliance, month_compute)].dropna().mean()
        gt_data[month_compute] = test_df['%s_%d' %(appliance, month_compute)]
        print pred[month_compute], gt_data[month_compute]
        error[month_compute] = np.abs(gt_data[month_compute]-pred[month_compute])*100/(gt_data[month_compute])

o = {'pred_df':pd.Series(pred),'gt_df':pd.Series(gt_data),'error':pd.Series(error)}
pickle.dump(o, open(os.path.join(store_path, filename),'w'))



