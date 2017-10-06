import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
from gini import gini_xgb

path = '~/Desktop/kaggle/Porto/'

df = pd.read_csv(path+'train.csv',index_col='id')

df = df.reset_index(drop=True)

features = list(df.columns)
target = 'target'
features.remove(target)

X = np.array(df[features])
y = df[target]

xgb = XGBClassifier(max_depth=4, 
                    learning_rate=0.05, 
                    n_estimators=1000, 
                    objective='binary:logistic',
                    nthread=-1, 
                    gamma=0, 
                    subsample=0.8,
                    colsample_bytree=0.8,  
                    #scale_pos_weight=30, 
                    missing=None)

kf = KFold(n_splits=3)
for train_index, test_index in kf.split(X):
    train_X, test_X = X[train_index], X[test_index]
    train_y, test_y = y[train_index], y[test_index]
    xgb.fit(train_X, train_y, 
            eval_set=[(train_X,train_y),(test_X,test_y)], 
            eval_metric=gini_xgb,
            early_stopping_rounds=10,
            verbose=10)
