#Negative words insertion use case:

from benchmark_methods_stats import (
    c2st,
    mmd_test,
    random_projection_test,
    adaptive_test_high_dim,
    chen_qin
)
from RRPerm import RRPerm
import autotst

def benchmark_methods(df1, df2, B = 100):
    n1 = df1.shape[0]
    n2 = df2.shape[0]
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
    for i in range(B):
        Y_res1 = Y1.reshape(-1, 1).astype(float)
        Y_res2 = Y2.reshape(-1, 1).astype(float)#(n2, 1)
        df1_concat = np.concatenate([df_batch1, Y_res1], axis = 1).astype(float)
        df2_concat = np.concatenate([df_batch2, Y_res1], axis = 1).astype(float)
        df1_X_concat = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
        df2_X_concat = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
        df1_sample = df1_concat[np.random.choice(np.arange(len(df1_concat)), size = len(df1_concat), replace = True), :].astype(float)
        df2_sample = df2_concat[np.random.choice(np.arange(len(df2_concat)), size = len(df2_concat), replace = True), :].astype(float)
        X = np.concatenate([df1_sample[:, :(df1_sample.shape[1] - 1)],
            df2_sample[:, :(df2_sample.shape[1] - 1)]]).astype(float)
        Y = np.concatenate([df1_sample[:, df1_sample.shape[1] - 1],
            df2_sample[:, df2_sample.shape[1] - 1]]).astype(float)
        pval_rrperm[i] = pval
        pval_xu16[i], _ = adaptive_test_high_dim(
            df1_sample, df2_sample, n_perm = n_perm)
        pval_chenqin[i] = chen_qin(df1_sample, df2_sample)
        pval_miles16[i] = random_projection_test(df1_sample, df2_sample)
        pval_c2st[i] = c2st(X, Y, B = c2st_perm_b)
        pval_miles

#今晚把所有方法写完 -- benchmark

def add_negative_comments_start(text_orig, n_negative_words, negative_words_list):
    text_orig_split = text_orig.split(" ")
    words_insert = random.sample(list(negative_words_list), n_negative_words)
    return " ".join(words_insert + text_orig_split)



df2["tweet_4"] = df2["tweet_text"].apply(lambda x: add_negative_comments_start(x, 4, neg_words))
df2_4word = df2["tweet_4"].apply(lambda X: bert_base_embedding(X, n_dim = 768))
df2_4word.to_csv("df2_4word_bert_.csv")

#Input is the dataset with columns with respect to:
def add_negative_comments_data(df, text_col, text_orig, n_negative_words, negative_words_list,
    output_col):
    df = pd.read_csv('')
    df[output_col] = df[text_col].apply(lambda X: 
        words_insert = random.sample(list(negative_words_list), n_negative_words, seed = seed)
        add_negative_comments_start(x, n_negative_words, negative_words_list))




























