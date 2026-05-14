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
from RFPerm import RFPerm
from RRPerm import RRPerm
from cfperm import *

'''
Benchmark a variety of different methods for comparisons:

'''
func_config = {
'n_perm_cf': ,
'n_iter_cs_tst': 200,
'n_perm_rf': 5000,
'kernel': 'rbf'
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
    n_perm_cs_tst = config.get('n_iter_cs_tst')
    #Initiate the p-value lists, for the two sample testing procedure, run on the joint distribution as well as the covariate distributions
    p_val_rrperm =  [0.0] * B
    p_val_miles16 = [0.0] * B
    p_val_xu16 = [0.0] * B
    p_val_chen10 = [0.0] * B
    p_val_chen14 = [0.0] * B
    p_val_c2st = [0.0] * B
    p_val_kmmd = [0.0] * B
    p_val_sri = [0.0] * B
    p_val_pan14 = [0.0] * B
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
        p_val_rrperm[i] = RRPerm()
        p_val_miles16_joint[i] = random_projection_test(df1_sample, df2_sample)
        p_val_kmmd_joint[i] = kernel_MMD(df1_sample, df2_sample, kernel = kernel)
        p_val_chen14[i] = chen_qin(df1_sample, df2_sample)
        p_val_chen10[i] = chen_li14(df1_sample, df2_sample)
    #Reformulate them back to a DataFrame:
    out_df = pd.DataFrame({})
    return out_df


















