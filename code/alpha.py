np.random.seed(42)
appliance_df = df.ix[all_homes]
if appliance=="hvac":
    start, stop=5, 11
else:
    start, stop=1, 13

test_homes = [home]
train_homes = appliance_df[~appliance_df.index.isin([home])].index
print test_homes, train_homes


#all_home_appliance = deepcopy(all_homes)
#all_home_appliance[appliance] = train_homes

# Cross validation on inner loop to find best feature, K
train_size = len(train_homes)
l = LeaveOneOut(train_size)
out = OrderedDict()
for cv_train, cv_test in l:
    print cv_test

    cv_train_home=appliance_df.ix[train_homes[cv_train]]
    cv_test_home = appliance_df.ix[train_homes[cv_test]]
    test_home_name = cv_test_home.index.values[0]
    #print cv_test_home, cv_train_home
    out[test_home_name]={}


    # Summing up energy across start to stop to get Y to learn optimum feature on
    Y = cv_train_home[['%s_%d' %(appliance, i) for i in range(start, stop)]].sum(axis=1).values
    forest = ExtraTreesRegressor(n_estimators=250,
                          random_state=0)
    forest.fit(cv_train_home[feature_map[feature]], Y)
    importances = forest.feature_importances_
    #print importances, feature_map[feature]
    indices = np.argsort(importances)[::-1]
    #print indices

    # Now varying K and top-N features

    for K in range(K_min, K_max):
        out[test_home_name][K]={}
        for top_n in range(F_min,F_max):
            out[test_home_name][K][top_n]=[]
            top_n_features = cv_train_home[feature_map[feature]].columns[indices][:top_n]
            #print top_n_features, top_n

            # Now fitting KNN on this
            for month in range(start, stop):
                clf = KNeighborsRegressor(n_neighbors=K)
                clf.fit(cv_train_home[top_n_features], cv_train_home['%s_%d' %(appliance, month)])
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
            #print pred, gt, error
            mean_error = error.squeeze()
            #print pred, gt, mean_error

            temp[h]=mean_error
        ac = pd.DataFrame(temp).T.median().mean()
        print ac, pd.DataFrame(temp).median()

        accur[K][top_n] = ac
#return accur
accur_df = pd.DataFrame(accur)
print accur_df
accur_min = accur_df.min().min()
print accur_min
max_ac_df = accur_df[accur_df==accur_min]
F_best = cv_train_home[feature_map[feature]].columns[indices][:max_ac_df.mean(axis=1).dropna().index.values[0]].tolist()
K_best = max_ac_df.mean().dropna().index.values[0]
