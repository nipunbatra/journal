import pickle

import numpy as np
import pandas as pd

from code.degree_days import  dd
from code.regional_average_contribution import  contribution

out_overall = pickle.load(open('../data/input/all_regions.pkl','r'))

import os

#train_region, test_region, test_home, appliance, transform, K, num_homes, NUM_TRIALS = sys.argv[1:]

train_region="SanDiego"
test_region="Austin"
appliance="hvac"
transform="DD-percentage"
NUM_TRIALS=1
num_homes=0
K=3
min_num_features=20
out_pred = {}
out_count = {}
rs = 0
np.random.seed(rs)

train_df = out_overall[train_region]
test_df = out_overall[test_region]
#test_df=test_df[(test_df.full_agg_available==1)&(test_df.md_available==1)]
if appliance=="hvac":
    start, stop=5, 11
else:
    start, stop=1, 13
#test_df =test_df.ix[test_df[['%s_%d' %(appliance,month) for month in range(start,stop)]].dropna().index]
for test_home in test_df.index[:]:
    out_pred[test_home]={}
    print test_home



    test_df_used_for_train = test_df.sample(num_homes)
    #print test_df_used_for_train

    train_dd = pd.DataFrame(dd[train_region])
    test_dd = pd.DataFrame(dd[test_region])

    median_aggregate = {}
    for region in [train_region, test_region]:
        median_aggregate[region] = {}
        for month in range(1, 13):
            median_aggregate[region][month] = out_overall[region]['aggregate_'+str(month)].median()

    median_aggregate_df = pd.DataFrame(median_aggregate)


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
        return com_f

    def find_distance_train_test(df_train, home_1, home_2, df_test, home_test, featureset_train, featureset_max):
        f_test = df_test.ix[home_test][featureset_max].dropna()
        com_f =  np.intersect1d(f_test.index, featureset_train)
        if len(com_f):
            is_common = True
        else:
            is_common = False
            return is_common, None

        if len(com_f):
            a = np.linalg.norm(df_train.ix[home_1][com_f]- df_test.ix[home_test][com_f])
            b = np.linalg.norm(df_train.ix[home_2][com_f]- df_test.ix[home_test][com_f])
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

    if transform in ["None","None-percentage"]:
        pass
    elif transform in ["DD","DD-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(1, 13):
            # index on 0, 11
            train_dd_month = train_dd.ix[month-1]['Total']
            test_dd_month = test_dd.ix[month-1]['Total']
            train_df['hvac_%d' % month] = train_df_copy['hvac_%d' % month] * test_dd_month*1. / train_dd_month

            #New aggregate will be removing old HVAC and adding new HVAC!
            train_df['aggregate_%d' %month] = train_df_copy['aggregate_%d' %month] - train_df_copy['hvac_%d' % month] + train_df['hvac_%d' % month]
    elif transform in ["median-aggregate","median-aggregate-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(1,13):
            median_month = median_aggregate_df.ix[month]
            cols_to_transform = [x for x in train_df.columns if "_"+str(month) in x]
            train_df[cols_to_transform] = train_df_copy[cols_to_transform] * median_month[test_region] / median_month[train_region]

    elif transform in ["regional","regional-percentage"]:
        train_df_copy = train_df.copy()
        for month in range(1, 13):

            # index on 0, 11
            if month in range(4,11):
                mode='Cooling'
            else:
                mode='Heating'

            train_dd_month = contribution[train_region][mode]['hvac']
            test_dd_month = contribution[test_region][mode]['hvac']

            train_df['hvac_%d' % month] = train_df_copy['hvac_%d' % month] * test_dd_month*1. / train_dd_month

            #New aggregate will be removing old HVAC and adding new HVAC!
            train_df['aggregate_%d' %month] = train_df_copy['aggregate_%d' %month] - train_df_copy['hvac_%d' % month] + train_df['hvac_%d' % month]



    elif transform=="DD-fridge":
        train_df_copy = train_df.copy()
        fridge_model = pickle.load(open('../data/input/SanDiego_fridge_dd_coef.pkl','r'))
        for month in range(1, 13):
            # index on 0, 11

            train_cdd_month = train_dd.ix[month-1]['Cooling']
            test_cdd_month = test_dd.ix[month-1]['Cooling']
            for fridge_home, fridge_home_model in fridge_model.iteritems():
                train_df.loc[fridge_home,'fridge_%d' % month] = fridge_home_model['baseline']+fridge_home_model['cdd']*test_cdd_month
                train_df.loc[fridge_home,'aggregate_%d' %month] = train_df_copy.ix[fridge_home]['aggregate_%d' %month] - train_df_copy.ix[fridge_home]['fridge_%d' % month] + train_df.ix[fridge_home]['fridge_%d' % month]

    overall_df = pd.concat([train_df, test_df_used_for_train, test_df]).drop_duplicates()

    normalised_df = normalise(overall_df)

    train_normalised_df = pd.concat([normalised_df.ix[train_df.index], normalised_df.ix[test_df_used_for_train.index]])
    test_normalised_df = normalised_df.ix[test_df.index].drop_duplicates()


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
        month_start, month_end = 1, 13

    #for month_compute in range(month_start, month_end):
    for month_compute in range(month_start, month_end):
        out_pred[test_home][month_compute] = 0
        out_count[month_compute] = 0

        num_features_all = {}
        ineq_dict = {}

        num_features_all[appliance] = {}
        ineq_dict[appliance] = {}

        #num_features_all[appliance][month_compute] = {}
        ineq_dict[appliance][month_compute] = {}

        candidate_homes = train_normalised_df['%s_%d' %(appliance, month_compute)].dropna().index.values
        # Removing the actual test home!
        candidate_homes = np.array(np.setdiff1d(candidate_homes, test_home))


        #num_features_all[appliance][month_compute][test_home] = defaultdict(int)
        from collections import defaultdict
        import pandas as pd
        co = defaultdict(int)
        store_path = '../../../output/output/ineq_cross_subset_kdd_%d/%s_%s_%s_%s_%d_%d_%d.pkl' %(NUM_TRIALS, train_region,
                                                                            test_region,
                                                                            transform,
                                                                            appliance,
                                                                            month_compute,
                                                                            test_home, K)
        #print store_path
        if os.path.exists(store_path):
            print "already exists"
            #continue


        if not np.isnan(test_normalised_df.ix[test_home]['%s_%d' %(appliance, month_compute)]):
            # We need to predict this value!
            # Find candidate set, train homes which have not null for this month
            # Now find features on pairs of homes in candidate homes
            for a,b in combinations(candidate_homes, 2):
                com_features = find_com_features_train(train_normalised_df, a, b, f_all)

                if len(com_features)>min_num_features:
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
            #print train_df.ix[ranks[:K]].index
            if "percentage" in transform:
                mean_proportion = (train_normalised_df.ix[ranks[:K]]['%s_%d' %(appliance, month_compute)]/ overall_df.ix[ranks[:K]]['aggregate_%d' %(month_compute)]).mean()

                pred = test_df.ix[test_home]['aggregate_%d' %month_compute]*mean_proportion

            else:
                pred = train_df.ix[ranks[:K]]['%s_%d' %(appliance, month_compute)].dropna().mean()
            out_pred[test_home][month_compute]+= pred
            out_count[month_compute]+= 1
            gt = test_df.ix[test_home]['%s_%d' %(appliance, month_compute)]
            print pred, gt



        else:
            # No need to predict
            pass

pred_df = pd.DataFrame(out_pred).T
gt_df = test_df[['%s_%d' %(appliance, month_compute) for month_compute in range(month_start, month_end)]]
gt_df = gt_df.ix[pred_df.index]
gt_df.columns = range(month_start, month_end)
error_df = (pred_df-gt_df).abs().div(gt_df)

"""
for month_in_pred in pred_df.index:
    store_path = '../../../output/output/ineq_cross_subset_kdd_%d/%d_%s_%s_%s_%s_%d_%d_%d.pkl' %(NUM_TRIALS, num_homes,
                                                                                          train_region,
                                                                            test_region,
                                                                            transform,
                                                                            appliance,
                                                                            month_in_pred,
                                                                            test_home, K)
    pickle.dump(pred_df.ix[month_in_pred], open(store_path,'w'))

"""
