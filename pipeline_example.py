#Testing pipeline, we take the bias finder data for example:
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
from scipy.stats import f, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from benchmark_methods_distributionshift_testing import *


'''
Illustration for the biasfinder dataset(with both concept drift and covariate shift exists)
'''
model_registry = default_model_registry(
  ntree = 150,
  ridge_alpha = 0.25,
  nthread = 1, maxit = 200, max_depth = 5,
  gamma = 0.25, eta = 0.15, mlp_hidden_size = 4,
  mlp_decay = 1e-5, mlp_max_iter = 500, mlp_trace = False,
  mlp_max_coef_reg = 10000, mlp_max_coef_clf = 10000,
  warn_xgb_labels = True, positive_class = 1
)
func_config = {
    'n_perm_cf': 150,
    'n_perm_rf': 150,
    'n_perm_drperm': 150,
    'n_perm_rrperm': 150,
    'outcome_model_causal': 'rf_regressor',
    'ps_model': '',
    'kernel': 'rbf',
    'n_conformal': 150,
    'ps_model': 'logistic_classifier'
}
df1 = pd.read_csv("datasets/train_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
df2 = pd.read_csv("datasets/test_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
df1 = df1.sample(500)
df2 = df2.sample(500)
df1_X = df1.iloc[:,:(df1.shape[1]-1)]
df2_X = df2.iloc[:,:(df2.shape[1]-1)]
df1['sentiment'] = df1['class'].apply(lambda X: 1 if X == 'sarc' else 0)
Y1 = np.asarray(df1['sentiment']).astype(float)
Y2 = np.asarray(df2['sentiment']).astype(float)
df_exist = np.column_stack([df1_X, Y1])
df_new = np.column_stack([df2_X, Y2])

result = benchmark_all_method(df1, df2, config = func_config)












