import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd

from degree_days import dd
from regional_average_contribution import contribution
from sklearn.cross_validation import KFold

def solve_ilp(inequalities, time_limit=50):
        from collections import defaultdict
        import pandas as pd
        co = defaultdict(int)
        for ineq in inequalities:
            lt = ineq[0]
            gt = ineq[1]
            co[lt] -= 1
            co[gt] += 1
        co_ser = pd.Series(co)
        co_ser.sort()

        return co_ser.index.values.tolist()


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
    for col in ['area', 'num_occupants', 'house_num_rooms', 'ratio_min_max',
                'skew', 'kurtosis', 'variance', 'difference_ratio_min_max', 'p_25',
                'p_50', 'p_75']:
        new_df[col] = scale_0_1(df[col])
    return new_df


# Find not null set of common features
def find_com_features_train(df, home_1, home_2, featureset_max):
    f_1 = df.ix[home_1][featureset_max].dropna()
    f_2 = df.ix[home_2][featureset_max].dropna()
    com_f = np.intersect1d(f_1.index, f_2.index)
    return com_f


def find_distance_train_test(df_train, home_1, home_2, df_test, home_test, featureset_train, featureset_max):
    f_test = df_test.ix[home_test][featureset_max].dropna()
    com_f = np.intersect1d(f_test.index, featureset_train)
    if len(com_f):
        is_common = True
    else:
        is_common = False
        return is_common, None

    if len(com_f):
        a = np.linalg.norm(df_train.ix[home_1][com_f] - df_test.ix[home_test][com_f])
        b = np.linalg.norm(df_train.ix[home_2][com_f] - df_test.ix[home_test][com_f])
        if a <= b:
            order = [home_1, home_2]
        else:
            order = [home_2, home_1]
        return is_common, {'order': order,
                           'num_f': len(com_f),
                           'dist_a': a,
                           'dist_b': b,
                           'f': com_f}


def transform_data(train_df, transform):
    if transform in ["None", "None-percentage"]:
        train_df_copy = train_df.copy()
    elif transform in ["DD", "DD-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(5, 11):
            # index on 0, 11
            train_dd_month = train_dd.ix[month - 1]['Total']
            test_dd_month = test_dd.ix[month - 1]['Total']
            train_df['hvac_%d' % month] = train_df_copy['hvac_%d' % month] * test_dd_month * 1. / train_dd_month

            # New aggregate will be removing old HVAC and adding new HVAC!
            train_df['aggregate_%d' % month] = train_df_copy['aggregate_%d' % month] - train_df_copy[
                'hvac_%d' % month] + train_df['hvac_%d' % month]
    elif transform in ["median-aggregate", "median-aggregate-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(1, 13):
            median_month = median_aggregate_df.ix[month]
            cols_to_transform = [x for x in train_df.columns if "_" + str(month) in x]
            train_df[cols_to_transform] = train_df_copy[cols_to_transform] * median_month[test_region] / median_month[
                train_region]

    elif transform in ["regional", "regional-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(1, 13):

            # index on 0, 11
            if month in range(4, 11):
                mode = 'Cooling'
            else:
                mode = 'Heating'

            train_dd_month = contribution[train_region][mode]['hvac']
            test_dd_month = contribution[test_region][mode]['hvac']

            train_df['hvac_%d' % month] = train_df_copy['hvac_%d' % month] * test_dd_month * 1. / train_dd_month

            # New aggregate will be removing old HVAC and adding new HVAC!
            train_df['aggregate_%d' % month] = train_df_copy['aggregate_%d' % month] - train_df_copy[
                'hvac_%d' % month] + train_df['hvac_%d' % month]



    elif transform == "DD-fridge":
        train_df_copy = train_df.copy()
        fridge_model = pickle.load(open('../data/input/SanDiego_fridge_dd_coef.pkl', 'r'))
        for month in range(1, 13):
            # index on 0, 11

            train_cdd_month = train_dd.ix[month - 1]['Cooling']
            test_cdd_month = test_dd.ix[month - 1]['Cooling']
            for fridge_home, fridge_home_model in fridge_model.iteritems():
                train_df.loc[fridge_home, 'fridge_%d' % month] = fridge_home_model['baseline'] + fridge_home_model[
                                                                                                     'cdd'] * test_cdd_month
                train_df.loc[fridge_home, 'aggregate_%d' % month] = train_df_copy.ix[fridge_home][
                                                                        'aggregate_%d' % month] - \
                                                                    train_df_copy.ix[fridge_home]['fridge_%d' % month] + \
                                                                    train_df.ix[fridge_home]['fridge_%d' % month]
    return train_df

train_region, test_region, appliance, transform, K, num_homes, num_trials, min_common_features = sys.argv[1:]
K = int(K)
num_homes = int(num_homes)
num_trials = int(num_trials)
min_common_features = int(min_common_features)

out_overall = pickle.load(open('../data/input/all_regions.pkl', 'r'))

train_region_train_df = out_overall[train_region]
df = out_overall[test_region]
#df = df[(df.full_agg_available == 1) & (df.md_available == 1)]
if appliance=="hvac":
    start, stop=5, 11
else:
    start, stop=1,13
appliance_df = df.ix[df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
all_homes_test_region = appliance_df.index

# Doing 5 fold cross validation on all homes
kf = KFold(len(all_homes_test_region), n_folds=5)

out = {}

for cv_loop_index, (train_index, test_index) in enumerate(kf):
    out[cv_loop_index]={}
    print cv_loop_index, "CV"

    # Set of homes for which we want to make the predictions
    test_region_test_df = appliance_df.ix[all_homes_test_region[test_index]]

    # Set of homes which we can use from the train
    test_region_train_df = appliance_df.ix[all_homes_test_region[train_index]]

    # Now, from the set of test_region_train_homes, we will choose `num_homes`
    # and add to the train set
    for trial in range(num_trials):
        print trial, "TRIAL", "\n*"*20
        out[cv_loop_index][trial]={}
        #test_region_train_subset_homes_idx = np.random.choice(len(test_region_train_df), num_homes, replace=False)
        #test_region_train_subset_homes = test_region_train_df.ix[test_region_train_df.index[test_region_train_subset_homes_idx]]

        train_dd = pd.DataFrame(dd[train_region])
        test_dd = pd.DataFrame(dd[test_region])

        median_aggregate = {}
        for region in [train_region, test_region]:
            median_aggregate[region] = {}
            for month in range(1, 13):
                median_aggregate[region][month] = out_overall[region]['aggregate_' + str(month)].median()

        median_aggregate_df = pd.DataFrame(median_aggregate)

        start_month, end_month = 1, 12
        agg_features = np.hstack([['aggregate_' + str(month) for month in range(start_month, end_month + 1)],
                                  'ratio_min_max', 'difference_ratio_min_max', 'p_25', 'p_50', 'p_75', 'skew', 'kurtosis'])
        md_features = ['area', 'house_num_rooms']
        features = {'md_agg': np.hstack([
            agg_features,
            md_features
        ]).tolist()}

        f_all = features['md_agg']

        train_region_train_df_copy = train_region_train_df.copy()

        train_region_subset_homes_idx = np.random.choice(len(train_region_train_df), num_homes, replace=False)
        train_region_subset_homes = train_region_train_df.ix[train_region_train_df.index[train_region_subset_homes_idx]]


        # Transforming the train data from the train region
        train_region_train_df_transformed = transform_data(train_region_subset_homes, transform)

        # Now, creating a total train_df by combining data from same region and transformed from other region
        overall_train_df = train_region_train_df_transformed
        #overall_train_df = pd.concat([train_region_train_df_transformed, test_region_train_subset_homes])

        combined_train_test_df_for_normalisation = normalise(pd.concat([overall_train_df, test_region_test_df]))
        overall_train_df_normalised = combined_train_test_df_for_normalisation.ix[overall_train_df.index]
        test_region_test_df_normalised = combined_train_test_df_for_normalisation.ix[test_region_test_df.index]

        if appliance == "hvac":
            month_start, month_end = 5, 11
        else:
            month_start, month_end = 1, 13

        for test_home in test_region_test_df_normalised.index:

            out[cv_loop_index][trial][test_home]={}
            #Now, we will predict for each month for each test home
            for month_compute in range(month_start, month_end):

                num_features_all = {}
                ineq_dict = {}

                num_features_all[appliance] = {}
                ineq_dict[appliance] = {}

                # num_features_all[appliance][month_compute] = {}
                ineq_dict[appliance][month_compute] = {}

                candidate_homes = overall_train_df_normalised['%s_%d' % (appliance, month_compute)].dropna().index.values


                from collections import defaultdict
                import pandas as pd

                co = defaultdict(int)
                if not np.isnan(test_region_test_df_normalised.ix[test_home]['%s_%d' % (appliance, month_compute)]):
                    # We need to predict this value!
                    # Find candidate set, train homes which have not null for this month
                    # Now find features on pairs of homes in candidate homes


                    count_c = 0
                    for a, b in combinations(candidate_homes, 2):
                        com_features = find_com_features_train(overall_train_df_normalised, a, b, f_all)

                        if len(com_features) > min_common_features:
                            count_c += 1
                            # Consider a,b
                            is_common, d = find_distance_train_test(overall_train_df_normalised, a, b, test_region_test_df_normalised, test_home,
                                                                    com_features, f_all)

                            if is_common:
                                # Common between train and test. Can add this pair to inequalities
                                ineq = d['order']
                                lt = ineq[0]
                                greater_than = ineq[1]
                                co[lt] -= 1
                                co[greater_than] += 1
                                # num_features_all[appliance][month_compute][test_home][d['num_f']]+= 1


                    co_ser = pd.Series(co)
                    co_ser.sort()
                    ranks = co_ser.index.values.tolist()
                    if "percentage" in transform:
                        mean_proportion = (
                            overall_train_df.ix[ranks[:K]]['%s_%d' % (appliance, month_compute)] / overall_train_df.ix[ranks[:K]][
                                'aggregate_%d' % (month_compute)]).mean()

                        pred = test_region_test_df.ix[test_home]['aggregate_%d' % month_compute] * mean_proportion

                    else:
                        pred = overall_train_df.ix[ranks[:K]]['%s_%d' % (appliance, month_compute)].dropna().mean()
                    gt = test_region_test_df.ix[test_home]['%s_%d' % (appliance, month_compute)]
                    error = np.abs(gt-pred)
                    error_percentage = error/gt
                    out[cv_loop_index][trial][test_home][month_compute] = error_percentage
                    print error_percentage, test_home, month_compute, trial, cv_loop_index

out_path = os.path.expanduser("~/output/unified/unified_num_homes_train_cv/")
if not os.path.exists(out_path):
    print "here"
    os.makedirs(out_path)
import pickle
filename = os.path.join(out_path, "%s_%d_%d_%s_%d.pkl" %(appliance, num_homes, min_common_features, transform, K))
pickle.dump(out, open(filename,'w'))
