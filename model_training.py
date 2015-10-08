from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import cross_validation

import numpy as np
from numpy import genfromtxt, savetxt

def main():
    #create the training & test sets, skipping the header row with [1:]
    dataset = genfromtxt('trainProcessed.csv', delimiter='::')[1:]
    columnsToIgnoreTrain = [5, 6, 17, 18]
    columnsInTrain = [i for i in range(np.shape(dataset)[1]) if i not in columnsToIgnoreTrain]
    target = dataset[:,5]
    train = dataset[:,columnsInTrain]
    test_raw = genfromtxt('testProcessed.csv', delimiter='::')[1:]
    columnsToIgnoreTest = [1, 16, 17]
    columnsInTest = [i for i in range(np.shape(test_raw)[1]) if not i in columnsToIgnoreTest]
    test = test_raw[:,columnsInTest]

    gb = GradientBoostingRegressor(n_estimators=700)
    gb.fit(train, target)
    rf1 = RandomForestRegressor(n_estimators=100, n_jobs=4)
    rf2 = RandomForestRegressor(n_estimators=150, max_features=6, n_jobs=4)
    rf3 = RandomForestRegressor(n_estimators=150, max_features=7, n_jobs=4)
    rf3.fit(train, target)
    scores1 = cross_validation.cross_val_score(gb, train, target, cv=5)
    scores2 = cross_validation.cross_val_score(rf2, train, target, cv=5)
    scores3 = cross_validation.cross_val_score(rf3, train, target, cv=5)
    cv_prediction = cross_validation.cross_val_predict(rf2, train, target, cv=8)

    prediction = [[test_raw[index][1], x] for index, x in enumerate(rf3.predict(test))]

    savetxt('submission_sk_rf_150_7_1.csv', prediction, delimiter=',', fmt='%d,%f',
            header='Id,Sales', comments = '')

if __name__=="__main__":
    main()
