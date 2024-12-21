import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.combine import SMOTEENN
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import StackingClassifier
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler, LabelEncoder

train_df = pd.read_csv('fraudTrain.csv')
test_df = pd.read_csv('fraudTest.csv')
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
train_df = train_df.fillna('')

label_cols = ['merchant', 'category', 'first', 'last', 'gender', 'street']
le = LabelEncoder()
for col in label_cols:
    train_df[col] = le.fit_transform(train_df[col])

train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
train_df['Time'] = (train_df['trans_date_trans_time'] - train_df['trans_date_trans_time'].min()).dt.total_seconds()

scaler = StandardScaler()
train_df['Amount'] = scaler.fit_transform(train_df['amt'].values.reshape(-1, 1))

X = train_df[['Time', 'Amount'] + label_cols]
y = train_df['is_fraud']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote_enn = SMOTEENN(random_state=42)
X_train_balanced, y_train_balanced = smote_enn.fit_resample(X_train, y_train)

def optimize_xgboost(learning_rate, max_depth, n_estimators, gamma):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    model = XGBClassifier(learning_rate=learning_rate, 
                          max_depth=max_depth, 
                          n_estimators=n_estimators, 
                          gamma=gamma, 
                          random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return auc

param_space_xgb = {
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 12),
    'n_estimators': (50, 500),
    'gamma': (0, 5)
}

optimizer_xgb = BayesianOptimization(
    f=optimize_xgboost,
    pbounds=param_space_xgb,
    random_state=42
)

optimizer_xgb.maximize(init_points=5, n_iter=20)
best_params_xgb = optimizer_xgb.max['params']
print("Best XGBoost Hyperparameters:", best_params_xgb)

def optimize_lightgbm(learning_rate, max_depth, n_estimators, num_leaves):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    num_leaves = int(num_leaves)
    model = LGBMClassifier(learning_rate=learning_rate, 
                           max_depth=max_depth, 
                           n_estimators=n_estimators, 
                           num_leaves=num_leaves, 
                           random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return auc

param_space_lgb = {
    'learning_rate': (0.01, 0.2),
    'max_depth': (3, 12),
    'n_estimators': (50, 500),
    'num_leaves': (20, 100)
}

optimizer_lgb = BayesianOptimization(
    f=optimize_lightgbm,
    pbounds=param_space_lgb,
    random_state=42
)

optimizer_lgb.maximize(init_points=5, n_iter=20)
best_params_lgb = optimizer_lgb.max['params']
print("Best LightGBM Hyperparameters:", best_params_lgb)

def optimize_catboost(learning_rate, depth, iterations, l2_leaf_reg):
    depth = int(depth)
    iterations = int(iterations)
    model = CatBoostClassifier(learning_rate=learning_rate, 
                               depth=depth, 
                               iterations=iterations, 
                               l2_leaf_reg=l2_leaf_reg, 
                               silent=True, 
                               random_state=42)
    model.fit(X_train_balanced, y_train_balanced)
    y_pred = model.predict(X_test)
    auc = roc_auc_score(y_test, y_pred)
    return auc

param_space_cat = {
    'learning_rate': (0.01, 0.2),
    'depth': (3, 10),
    'iterations': (100, 1000),
    'l2_leaf_reg': (1, 10)
}

optimizer_cat = BayesianOptimization(
    f=optimize_catboost,
    pbounds=param_space_cat,
    random_state=42
)

optimizer_cat.maximize(init_points=5, n_iter=20)
best_params_cat = optimizer_cat.max['params']
print("Best CatBoost Hyperparameters:", best_params_cat)

xgb_best = XGBClassifier(**best_params_xgb, random_state=42)
lgb_best = LGBMClassifier(**best_params_lgb, random_state=42)
cat_best = CatBoostClassifier(**best_params_cat, silent=True, random_state=42)

stacking = StackingClassifier(estimators=[('XGB', xgb_best), 
                                           ('LGB', lgb_best), 
                                           ('Cat', cat_best)],
                              final_estimator=XGBClassifier(random_state=42))

stacking.fit(X_train_balanced, y_train_balanced)

y_pred = stacking.predict(X_test)
print(classification_report(y_test, y_pred))
print('ROC AUC Score:', roc_auc_score(y_test, y_pred))
