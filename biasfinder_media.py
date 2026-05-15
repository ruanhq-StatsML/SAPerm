#rrperm benchmark bias finder and media bias:

from rr_perm_0419 import _r_risk_on_W, _r_risk_perm_w_pvalue
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
os.chdir("/Users/heqiaoruan/Desktop/Desktop - heqiao’s MacBook Air/PhD/Research/permutationTestingCovariateShift")
df1 = pd.read_csv("train_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
df2 = pd.read_csv("test_biasfinder_bert_embedding.csv").drop(['Unnamed: 0'], axis = 1)
#running the RRPerm procedure after subsampling:

df1 = df1.sample(1000)
df1_X = df1.iloc[:,:(df1.shape[1]-1)]
df2_X = df2.iloc[:,:(df2.shape[1]-1)]
df1['sentiment'] = df1['class'].apply(lambda X: 1 if X == 'sarc' else 0)
Y1 = np.asarray(df1['sentiment'])
Y2 = np.asarray(df2.iloc[:, 768]).astype(float)
#Fitting the PCA on both df1 and df2:
from sklearn.decomposition import PCA
pca = PCA(n_components = 100, random_state = 2026)
z = np.vstack([df1_X, df2_X])
Z_pc = pca.fit_transform(z)
df_batch1 = Z_pc[:1000, :]
df_batch2 = Z_pc[1000:, :]
B = 50
pval_list = [0] * B
n1 = df_batch1.shape[0]
n2 = df_batch2.shape[0]
NUISANCE = 'linear'
W = np.concatenate([np.zeros(n1, dtype = int),
                    np.ones(n2, dtype = int)])
for i in range(B):
    Y_res1 = Y1.reshape(-1, 1).astype(float)
    Y_res2 = Y2.reshape(-1, 1).astype(float)
    df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
    df2_concat = np.concatenate([df_batch2, Y_res2], axis = 1).astype(float)
    df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)], 
                        df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
    Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
                        df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
    risk, _bias_arm, _, _, _, _ = _r_risk_on_W(
            X, Y, W, cf_seed = 2026 + i,
            tau_seed = 2024 + i,
            n_splits = 5,
            nuisance = NUISANCE,
            clip_wtilde = 0.005,
            rf_n_estimators = 50
        )
    pval, _, _, _, _, _, _, _ = _r_risk_perm_w_pvalue(
            X,
            Y,
            W,
            n_resamples=int(100),
            sim_seed=2025,
            n_splits=5,
            nuisance=NUISANCE,
            rf_n_estimators=50,
        )
    pval_list[i] = pval
    print(pval)
np.mean(np.asarray(pval_list) < 0.05)#biasfinder dataset:




from rr_perm_0419 import _r_risk_on_W, _r_risk_perm_w_pvalue
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
os.chdir("/Users/heqiaoruan/Desktop/Desktop - heqiao’s MacBook Air/PhD/Research/permutationTestingCovariateShift")
df = pd.read_csv("media_bias_with_embedding.csv").drop(['Unnamed: 0'], axis = 1)
#running the RRPerm procedure after subsampling:
df['label'] = df['label'].apply(lambda X: 1 if X == 'Biased' else 0)
df1 = df[df['gender'] == 'Male']
df2 = df[df['gender'] == 'Female']
df1 = df1.sample(1000)
df2 = df2.sample(1000)
df1_X = df1.iloc[:,:768]
df2_X = df2.iloc[:,:768]
Y1 = np.asarray(df1['label']).astype(float)
Y2 = np.asarray(df2['label']).astype(float)
#Fitting the PCA on both df1 and df2:
from sklearn.decomposition import PCA
pca = PCA(n_components = 100, random_state = 2026)
z = np.vstack([df1_X, df2_X])
Z_pc = pca.fit_transform(z)
df_batch1 = Z_pc[:1000, :]
df_batch2 = Z_pc[1000:, :]
B = 50
pval_list = [0] * B
n1 = df_batch1.shape[0]
n2 = df_batch2.shape[0]
NUISANCE = 'linear'
W = np.concatenate([np.zeros(n1, dtype = int),
                    np.ones(n2, dtype = int)])
for i in range(B):
    Y_res1 = Y1.reshape(-1, 1).astype(float)
    Y_res2 = Y2.reshape(-1, 1).astype(float)
    df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
    df2_concat = np.concatenate([df_batch2, Y_res2], axis = 1).astype(float)
    df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)], 
                        df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
    Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
                        df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
    risk, _bias_arm, _, _, _, _ = _r_risk_on_W(
            X, Y, W, cf_seed = 2026 + i,
            tau_seed = 2024 + i,
            n_splits = 5,
            nuisance = NUISANCE,
            clip_wtilde = 0.005,
            rf_n_estimators = 50
        )
    pval, _, _, _, _, _, _, _ = _r_risk_perm_w_pvalue(
            X,
            Y,
            W,
            n_resamples=int(100),
            sim_seed=2025,
            n_splits=5,
            nuisance=NUISANCE,
            rf_n_estimators=50,
        )
    pval_list[i] = pval
    print(pval)
np.mean(np.asarray(pval_list) < 0.05)




######################################################
import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
os.chdir("/Users/heqiaoruan/Desktop/Desktop - heqiao’s MacBook Air/PhD/Research/permutationTestingCovariateShift")
from rr_perm_0419 import _r_risk_on_W, _r_risk_perm_w_pvalue
df = pd.read_csv("media_bias_with_embedding.csv").drop(['Unnamed: 0'], axis = 1)
#running the RRPerm procedure after subsampling:
df['label'] = df['label'].apply(lambda X: 1 if X == 'Biased' else 0)
df1 = df[df['age'] >= 35]
df2 = df[df['age'] < 35]
df1 = df1.sample(1000)
df2 = df2.sample(1000)
df1_X = df1.iloc[:,:768]
df2_X = df2.iloc[:,:768]
Y1 = np.asarray(df1['label']).astype(float)
Y2 = np.asarray(df2['label']).astype(float)
#Fitting the PCA on both df1 and df2:
from sklearn.decomposition import PCA
pca = PCA(n_components = 100, random_state = 2026)
z = np.vstack([df1_X, df2_X])
Z_pc = pca.fit_transform(z)
df_batch1 = Z_pc[:1000, :]
df_batch2 = Z_pc[1000:, :]
B = 50
pval_list = [0] * B
n1 = df_batch1.shape[0]
n2 = df_batch2.shape[0]
NUISANCE = 'linear'
W = np.concatenate([np.zeros(n1, dtype = int),
                    np.ones(n2, dtype = int)])
for i in range(B):
    Y_res1 = Y1.reshape(-1, 1).astype(float)
    Y_res2 = Y2.reshape(-1, 1).astype(float)
    df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
    df2_concat = np.concatenate([df_batch2, Y_res2], axis = 1).astype(float)
    df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
    df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
    X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)], 
                        df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
    Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
                        df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
    risk, _bias_arm, _, _, _, _ = _r_risk_on_W(
            X, Y, W, cf_seed = 2026 + i,
            tau_seed = 2024 + i,
            n_splits = 5,
            nuisance = NUISANCE,
            clip_wtilde = 0.005,
            rf_n_estimators = 50
        )
    pval, _, _, _, _, _, _, _ = _r_risk_perm_w_pvalue(
            X,
            Y,
            W,
            n_resamples=int(100),
            sim_seed=2025,
            n_splits=5,
            nuisance=NUISANCE,
            rf_n_estimators=50,
        )
    pval_list[i] = pval
    print(pval)


np.mean(np.asarray(pval_list) < 0.05)


