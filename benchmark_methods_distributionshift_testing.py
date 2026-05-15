#Benchmark All comparison methods:
import numpy as np
from scipy.stats import f, norm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split
from benchmark_method_tst import (
    chen_qin, adaptive_test_high_dim,
    mmd_test, c2st,
    zwl_test, bai96_test,
    sri08_test, skk_test,
    random_projection_test,
    kernel_MMD
)
from CFPerm_vimp_combined import *
from RFPerm import RFPerm
from RRPerm import RRPerm
from cfperm import *
from RFPerm import * 
from model_registry_class import *
'''
Benchmark a variety of different methods for comparisons:

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

def L2(y_pred, y_target):
    return ((np.array(y_pred) - np.array(y_target))**2).tolist()

func_config = {
'n_perm_cf': 150,
'n_perm_drperm': 150,
'n_perm_rf': 5000,
'kernel': 'rbf',
'n_perm_rrperm': 150,
'outcome_model': 'rf_regressor',
'ps_model': 'logistic_classifier'
}
def benchmark_all_method(df1, df2, B, config = func_config):
    n1, p = df1.shape
    n2 = df2
    df_batch1 = np.asarray(df1[col_df1].to_numpy(dtype=float), dtype=float)
    df_batch2 = np.asarray(df2[col_df2].to_numpy(dtype=float), dtype=float)
    if df_batch1.ndim == 1:
        df_batch1 = df_batch1.reshape(-1, 1)
    if df_batch2.ndim == 1:
        df_batch2 = df_batch2.reshape(-1, 1)
    Y1 = df1[col_Y].values
    Y2 = df2[col_Y].values
    Y = np.concatenate([Y1, Y2])
    W = np.concatenate([np.zeros(n1, dtype = int),
                    np.ones(n2, dtype = int)])
    n_perm_cf = config.get("n_perm_cf")
    n_perm_rf = config.get('n_perm_rf')
    n_perm_drperm = config.get('n_perm_drperm')
    outcome_model = config.get('outcome_model_causal')
    kernel = config.get('kernel')
    n_perm_rrperm = config.get('n_perm_rrperm')
    ps_model = config.get('ps_model')
    #Initiate the p-value lists, for the two sample testing procedure, run on the joint distribution as well as the covariate distributions
    p_val_rrperm = np.zeros(B)
    p_val_miles16 = np.zeros(B)
    p_val_bai96 = np.zeros(B)
    p_val_xu16 = np.zeros(B)
    p_val_chen10 = np.zeros(B)
    p_val_chen14 = np.zeros(B)
    p_val_c2st = np.zeros(B)
    p_val_kmmd = np.zeros(B)
    p_val_zwl = np.zeros(B)
    p_val_zwlm = np.zeros(B)
    p_val_sri = np.zeros(B)
    p_val_pan14 = np.zeros(B)
    p_val_drperm = np.zeros(B)
    p_val_cfperm_loco = np.zeros(B)
    p_val_cfperm_permu = np.zeros(B)
    p_val_cfperm_grf = np.zeros(B)
    p_val_rfperm = np.zeros(B)
    p_val_autotst = np.zeros(B)
    p_val_skk = np.zeros(B)
    p_val_xgbperm = np.zeros(B)
    p_val_conformal = np.zeros(B)
    W = np.concatenate([np.zeros(n1, dtype = int), np.ones(n2, dtype = int)])
    #Resampling via the Bootstrap procedure:
    for i in range(B):
    	Y_res1 = Y1.reshape(-1, 1).astype(float)
        Y_res2 = Y2.reshape(-1, 1).astype(float)
        df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
        df2_concat = np.concatenate([df_batch2, Y_res2], axis = 1).astype(float)
        df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
        df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
        #Bootstrap version of the bootstrap sample:
        df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
        df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
        X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)], 
                            df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
        Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
                            df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
        p_val_rrperm[i] = RRPerm(X, Y, W, n_perm = n_perm_rrperm)
        p_val_cfperm_loco[i] = cfperm_feature_pvals(X, Y, W, vimp = 'loco', n_perm = n_perm_cf)
        p_val_cfperm_permu[i] = cfperm_feature_pvals(X, Y, W, vimp = 'permucate', n_perm = n_perm_cf)
        p_val_cfperm_grf[i] = cfperm_feature_pvals(X, Y, W, vimp = 'grf', n_perm = n_perm_cf)
        p_val_drperm[i] = DRPerm(X, Y, W, n_perm = n_perm_drperm)
        p_val_miles16[i] = random_projection_test(df1_X_concat, df2_X_concat)
        p_val_xu16[i] = adaptive_test_high_dim(df1_X_concat, df2_X_concat)
        p_val_kmmd[i] = kernel_MMD(df1_X_concat, df2_X_concat, kernel = kernel)
        p_val_chen14[i] = chen_qin(df1_X_concat, df2_X_concat)
        p_val_chen10[i] = chen_li14(df1_X_concat, df2_X_concat)
        p_val_pan14[i] = pan14(df1_X_concat, df2_X_concat)
        p_val_skk[i] = skk_test(df1_X_concat, df2_X_concat)
        p_val_sri[i] = sri08_test(df1_X_concat, df2_X_concat)
        p_val_bai96[i] = bai96_test(df1_X_concat, df2_X_concat)
        p_val_zwl[i] = zwl_test(df1_X_concat, df2_X_concat, order = 1)
        p_val_zwlm[i] = zwl_test(df1_X_concat, df2_X_concat, order = 2)
        p_val_c2st[i] = c2st(df1_X_concat, df2_X_concat)
        p_val_rfperm[i] = RFPerm(df1_sample, df2_sample, loss = L2, B = 200)
        p_val_xgbperm[i] = PermValTest(df1_sample, df2_sample, model_class = model_class, model_component = model_component_xgb)
        p_val_autotst[i] = autotst.AutoTST(df1_X_concat, df2_X_concat).p_value()
        p_val_conformal[i] = conformalNN(df1_sample, df2_sample)
    #Reformulate them back to a DataFrame:
    out_df = pd.DataFrame({
        'xu16': float(np.mean(p_val_xu16 < alpha)),
        'pan14': float(np.mean(p_val_pan14 < alpha)),
        'chen10': float(np.mean(p_val_chen10 < alpha)),
        'chen14': float(np.mean(p_val_chen14 < alpha)),
        'c2st': float(np.mean(p_val_c2st < alpha)),
        'sri': float(np.mean(p_val_sri < alpha)),
        'skk': float(np.mean(p_val_skk < alpha)),
        'zwl': float(np.mean(p_val_zwl < alpha)),
        'zwlm': float(np.mean(p_val_zwlm < alpha)),
        'bai96': float(np.mean(p_val_bai96 < alpha)),
        'rrperm': float(np.mean(p_val_rrperm < alpha)),
        'drperm': float(np.mean(p_val_drperm < alpha)),
        'miles16': float(np.mean(p_val_miles16 < alpha)),
        'kmmd': float(np.mean(p_val_kmmd < alpha)),
        'cfperm_loco': float(np.mean(p_val_cfperm_loco)),
        'cfperm_permu': float(np.mean(p_val_cfperm_permu)),
        'cfperm_grf': float(np.mean(p_val_cfperm_grf)),
        'conformal': float(np.mean(p_val_conformal)),
        'rfperm': float(np.mean(p_val_rfperm < alpha)),
        'xgbperm': float(np.mean(p_val_xgbperm < alpha)),
        'autotst': float(np.mean(p_val_autotst < alpha))
        })
    return out_df


















