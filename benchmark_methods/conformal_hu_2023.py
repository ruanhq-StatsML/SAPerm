#conformal testing benchmark:
import numpy as np
from scipy.stats import f, norm, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def centropy(p, y):
    p = np.clip(p, 1e-3, 1-1e-3)
    return np.mean(-y * np.log(p) - (1 - y) * np.log(1 - p))

def gerror(g1, g2):
    return np.sum(np.abs(g1/np.sum(g1) - g2/np.sum(g2)))

def getFinalStat(g1, g2, v1, v2):
    m1 = len(g1)
    m2 = len(g2)
    U = getUstat(g1, v1, v2)
    sigma1 = getAsyVar1(g1, v1, v2)['sigma']
    sigma2 = getAsyVar2(g2, v2, m1)['sigma']
    sigma_gm = np.sqrt(sigma1 * sigma2)
    sigma_hm = 2.0/(1/sigma1 + 1/sigma2)
    z1 = (U - 0.5)/np.sqrt(sigma1)
    z2 = (U - 0.5)/np.sqrt(sigma2)
    z_gm = (U - 0.5)/np.sqrt(sigma_gm)
    z_hm = (U - 0.5)/np.sqrt(sigma_hm)
    return {'U': U, 'z1': z1, 'z2': z2,
            'z_gm': z_gm, 'z_hm': z_hm,
            'sigma1': sigma1, 'sigma2': sigma2,
            'sigma_gm': sigma_gm, 'sigma_hm': sigma_hm}

def getUstat(g1, v1, v2):
    n1 = len(g1)
    n2 = len(v2)
    zeta = np.random.uniform(size = n2)
    ecdfVal = getEcdf1(v1, v2)
    ecdf1 = ecdfVal['ecdf1']
    inner_sum = np.array([zeta * np.mean(x == v2) for x in v1])
    u1 = np.mean(g1 * ((1 - ecdf1) + inner_sum))
    return(u1/np.mean(g1))


def getEcdf2(v):
    v_sorted = np.sort(v)
    ecdf1 = np.searchsorted(v_sorted, v, side = 'right')/len(v)
    ecdf2 = np.array([np.mean(v == x) for x in v])
    return ecdf1, ecdf2

def getEcdf1(v1, v2):
    v2_sorted = np.sort(v2)
    point1 = np.searchsorted(v2_sorted, v1, side = 'right')/len(v2)
    ecdf2 = np.array([np.mean(v2 == x) for x in v1])
    return ecdf1, ecdf2

def getUstat(g1, v1, v2):
    n1 = len(g1)
    n2 = len(v2)
    zeta = np.random.uniform(0, 1, n2)
    ecdf1 = getEcdf1


def sim_fun(x1, y1, x2, y2, n11, n12, n21, n22):
    n1 = n11 + n12
    n2 = n21 + n22
    index1 = np.random.choice(n1, size = n11, replace = False)
    index1_full = np.arange(n1)
    index1_comp = np.setdiff1d(index1_full, index1)
    x11 = x1[index1,:]
    x12 = x1[-index1,:]
    y11 = y1[index1]
    y12 = y1[-index1]
    index2 = np.random.choice(n2, size = n21, replace = False)
    index2_full = np.arange(n2)
    index2_comp = np.setdiff1d(index2_full, index2)
    g12_orac = np.ones(n12)
    g22_orac = np.ones(n22)
    v12_orac = np.ones(n12)
    v22_orac = np.ones(n22)
    temp = getFinalStat(g12_orac, g22_orac, v12_orac, v22_orac)
    u_orac = temp['U']
    var_orac1 = temp['sigma1']
    oracpvalue1 = norm.pdf(temp['z1'])
    var_orac2 = temp['sigma2']
    oracpvalue2 = norm.pdf(temp['z2'])
    oracpvalue_gm = norm.pdf(temp['z_gm'])


def getAsyVar2(g2, v2, m1):
    m2 = len(g2)
    ecdfVal = getEcdf2(v2)
    ecdf1 = ecdfVal['ecdf1']
    ecdf2 = ecdfVal['ecdf2']
    s1 = np.mean(g2 * (1 - ecdf1) ** 2) - 1/4
    s2 = np.mean(g2 * ecdf2**2)/3.0
    s3 = np.mean(g2 * (1 - ecdf1) * ecdf2)
    ss = s1 + s2 + s3
    ss = np.maximum(ss, 0)
    sigma1 = ss/m1 + (1/12)/m2
    sigma2 = (np.mean(g2) - 1)/m1
    sigma2 = np.maximum(sigma2, 0)
    sigma12 = (np.mean(g2 * (1 - ecdf1)) + np.mean(g2 * ecdf2)/2 - 1/2)/m1
    sigma = sigma1 + sigma2/4 - sigma12
    return {'sigma': sigma, 'sigma1': sigma1,
            'sigma2': sigma2, 'sigma12': sigma12}




def getAsyVar1(g1, v1, v2):
    m1 = len(g1)
    m2 = len(v2)
    ecdfVal = getEcdf1(v1, v2)
    ecdf1 = ecdfVal['ecdf1']
    ecdf2 = ecdfVal['ecdf2']
    ss = np.mean(g1**2 * (1 - ecdf1) ** 2) + np.mean(g1**2 * ecdf**2)/3 + 









def NNfun(x, z, xpre1, xpre2, n_epochs = 20, nnrep = 10, optimizer = 'sgd', clip_val = 1e-3, seed = 2026):
    n1 = x.shape[0]
    n2 = z.shape[0]
    prob1 = np.zeros((n1, nnrep))
    prob2 = np.zeros((n2, nnrep))
    for i in range(nnrep):
        seed_i = seed + i
        nn_model = MLPClassifier(
            hidden_layer_sizes = hidden_layers,
            activation = 'relu',
            solver = 'sgd',
            alpha = 0.0, 
            learning_rate_init = learn_rates,
            max_iter = n_epochs,
            random_state = seed,
            verbose = False,
            early_stopping = False
        	)
        nn_model.fit(x, z)
        prob1[:, i] = nn_model.preidct(xpre1)[:, 1]
        prob2[:, i] = nn_model.predict(xpre2)[:, 1]
    prob1_fit = np.mean(prob1, axis = 1)
    prob2_fit = np.mean(prob2, axis = 1)
    prob1_fit = np.clip(prob1_fit, clip_val, 1 - clip_val)
    prob2_fit = np.clip(prob2_fit, clip_val, 1 - clip_val)
    return {'prob1_fit': prob1_fit, 'prob2_fit': prob2_fit}


def KLR_cv(X, Y, xpre, ypre, lambdaseq, sigmaseq, n_splits = 5):
    lambda_len = len(lambdaseq)
    sigma_len = len(sigmaseq)
    centropy = np.zeros((lambda_len, sigma_len))
    ypre = ypre.astype(int)
    minerror = np.inf
    for i, lambda_now in range(lambda_len):
        for j, sigma in range(sigma_len):
            cv_data = KFold(n_splits = 5, shuffle = True, random_state = 24)
            pipe = Pipeline([
                ('rbf': RBFSampler()),
                ('lr': LogisticRegression())])
            param_grid = {
            'rbf__gamma': sigmaseq,
            'lr__C': 1/np.array(lambdaseq)
            }
            fit_klr = pipe.fit(X, Y)
            prob_pre = 1/(1 + np.exp(-K))
            prob_pre = np.minimum(np.maximum(prob_pre, 0.01), 0.99)
            centropy[i, j] = np.mean(-ypre * np.log(prob_pre) -
                (1 - ypre) * np.log(1 - prob_pre))
            if centropy[i, j] < minerror:
                minerror = centropy[i, j]
                iind = i
                jind = j 
                prob = prob_pre
    prob = np.minimum(np.maximum(prob, 0.01), 0.99)
    lambdas = lambdaseq[iind]
    sigma = sigmaseq[jind]
    return {
    'prob': prob,
    'lambda': lambdas,
    'sigma': sigma,
    'centropy': centropy
    }


def getCor(g1, v1, v2, gorac1, vorac1, vorac2):
    mean_hatg_minus_g = np.mean(g1 - gorac1)
    mean_abs_hatg_minus_g = np.mean(np.abs(g1 - gorac1))
    ecdfVal = getEcdf1(v1, v2)
    ecdf1 = ecdfVal['ecdf1']
    ecdf2 = ecdfVal['ecdf2']
    mean_hatd = np.mean(1 -  ecdf1 + ecdf2/2)
    var_hatg_minus_g = np.var(g1 - gorac1)
    second_hatd = np.mean(1 - ecdf1 + ecdf2/3)
    var_hatd = second_hatd - mean_hatd**2
    sceond_hatg_minus_g_hatd = np.mean((g1 - gorac1)*((1-ecd1)+ecdf2/2))
    e1 = sceond_hatg_minus_g_hatd - mean_hatg_minus_g * mean_hatd
    e2 = mean_hatg_minus_g
    e3 = np.mean(gorac1*((1-ecdf1)+ecdf2/2))
    rho1 = e1/np.sqrt(var_hatg_minus_g*var_hatg)
    rho2 = mean_hatg_minus_g/mean_abs_hatg_minus_g
    var_g = np.var(gorac1)
    rho3 = e3/np.sqrt(var_hatd * var_g)
    ecdfValorac = getEcdf1(vorac1, vorac2)
    ecdforac1 = ecdfValorac['ecdf1']
    ecdforac2 = ecdfValorac['ecdf2']
    e4 = np.mean(gorac1 * ((1 - ecdf1) + ecdf2/2)) - np.mean(gorac1 * ((1 - ecdforac1) + ecdforac2/2))
    return {
    'rho1': rho1, 'rho2': rho2,
    'rho3': rho3, 'e1': e1,
    'e2': e2, 'e3': e3,
    'e4': e4
    }
                

def LR(x1, x2, y1, y2):
    n1, p = x1.shape
    n2 = x2.shape[0]
    n = n1 + n2
    x = np.vstack((x1, x2))
    y = np.concatenate((y1, y2))
    xmean = np.tile(x.mean(axis = 0), (n, 1))
    ymean = np.mean(y)
    b = np.linalg.solve((x-xmean).T@(x-xmean), (x-xmean).T@(y-ymean))
    a = ymean - np.sum(b*x.mean(axis = 0))
    sigma2_res = np.mean((y - a - x@b)**2)
    xmean1 = np.tile(x1.mean(axis = 0), (n1, 1))
    xmean2 = np.tile(x2.mean(axis = 0), (n2, 1))
    ymean1 = np.mean(y1)
    ymean2 = np.mean(y2)
    X1tX1 = (x1-xmean1).T@(x1-xmean1)
    X2tX2 = (x2-xmean2).T@(x2-xmean2)
    X1tY1 = (x1-xmean1).T@(y1-ymean1)
    X2tY2 = (x2-xmean2).T@(y2-ymean2)
    b = np.linalg.solve(X1tX1 + X2tX2, X1tY1+X2tY2)
    a1 = ymean1 - np.sum(b * x1.mean(axis = 0))
    a2 = ymean2 - np.sum(b * x2.mean(axis = 0))
    sigma2_unres = (np.sum((y1 - a1 - x1@b)**2) + 
        np.sum((y2 - a2 - x2@b)**2))/(n1 + n2)
    stat = n * np.log(sigma2_res/sigma2_unres)
    pval = 1 - chi2.pdf(stat, 1)
    return pval


def sim_fun(x1, y1, x2, y2, n11, n12, n21, n22):
    n1 = n11 + n12
    n2 = n21 + n22
    index = np.random.choice(np.arange(n1), n11, replace = False)
    m_index = np.setdiff1d(np.arange(n1), index)
    x11 = x1[index,:]
    x12 = x1[m_index,:]
    y11 = y1[index]
    y12 = y1[m_index]
    index2 = np.random.choice(np.arange(n2), n12, replace = False)
    m_index = np.setdiff1d(np.arange(n2), inde2)
    x21 = x2[index, :]
    x22 = x2[m_index, :]
    y21 = y2[index]
    y22 = y2[m_index]
    g12_orac = np.ones(n12)
    g22_orac = np.ones(n22)
    v12_orac = np.ones(n12)
    v22_orac = np.ones(n22)
    temp = getFinalStat(g12_orac, g22_orac, v12_orac, v22_orac)
    u_orac = temp['U']
    var_orac1 = temp['sigma1']
    oracpvalue1 = norm.pdf(temp['z1'])
    var_orac2 = temp['sigma2']
    oracpvalue2 = norm.pdf(temp['z2'])
    oracpvalue_gm = norm.pdf(temp['z_gm'])
    oracpvalue_hm = norm.pdf(temp['z_hm'])
    label_fit_Y = np.concatenate((np.zeros(n11), np.ones(n21)))
    xy_fit = np.hstack((np.vstack(x11, x21), np.vstack(y11, y21)))
    fit_joint = LogisticRegression().fit(xy_fit, label_fit_Y)
    x_fit = np.vstack((x11, x21))
    fit_margi = LogisticRegression().fit(x_fit, label_fit_Y)
    prob_marginal = fit_margi.predict_proba(np.vstack((x12, x22)))[:, 1]
    prob_marginal = np.maximum(np.minimum(prob_marginal, 0.99), 0.01)
    g12_est_ll = prob_marginal[1:n12]/(1 - prob_marginal[1:n12])*n11/n21
    g22_est_ll = prob_marginal[(n21+1):(n12+n22)]/(1-prob.matginal[(n12+1):(n12+n22)])*n11/n21
    cerror12_ll_marginal = np.mean(prob_marginal[1:n12] > 0.5)
    cerror22_ll_marginal = np.mean()





def conformalNN(df1, df2, mc, alpha=0.05):
    """
    Python version of the R conformal_test function.
    
    Parameters
    ----------
    df1 : pandas.DataFrame
        First data frame. Last column is the response y, all previous columns are predictors.
    df2 : pandas.DataFrame
        Second data frame. Last column is the response y.
    mc : int
        Number of bootstrap replications.
    alpha : float, optional
        Significance level (default 0.05).
    
    Returns
    -------
    power : float
        Mean rejection rate across all tests.
    """
    p_X = df1.shape[1] - 1          # number of predictors
    n1 = len(df1)
    n2 = len(df2)
    # Pre-allocate result arrays
    llpvalue1 = np.zeros((mc, 3))
    nnpvalue1 = np.zeros((mc, 3))
    llpvalue2 = np.zeros(mc)
    nnpvalue2 = np.zeros(mc)
    llpvalue_gm = np.zeros((mc, 3))
    nnpvalue_gm = np.zeros((mc, 3))
    llpvalue_hm = np.zeros((mc, 3))
    nnpvalue_hm = np.zeros((mc, 3))
    centropy_ll_joint = np.zeros(mc)
    centropy_ll_marginal = np.zeros(mc)
    centropy_nn_joint = np.zeros(mc)
    centropy_nn_marginal = np.zeros(mc)
    error_ll = np.zeros(mc)
    error_nn = np.zeros(mc)
    cerror12_ll_joint = np.zeros(mc)
    cerror12_ll_marginal = np.zeros(mc)
    cerror12_nn_joint = np.zeros(mc)
    cerror12_nn_marginal = np.zeros(mc)
    cerror22_ll_joint = np.zeros(mc)
    cerror22_ll_marginal = np.zeros(mc)
    cerror22_nn_joint = np.zeros(mc)
    cerror22_nn_marginal = np.zeros(mc)
    for run in range(mc):
        df1_b = df1.sample(n=n1, replace=True, ignore_index=True)
        df2_b = df2.sample(n=n2, replace=True, ignore_index=True)
        x1 = df1_b.iloc[:, :p_X]
        y1 = df1_b.iloc[:, p_X]
        x2 = df2_b.iloc[:, :p_X]
        y2 = df2_b.iloc[:, p_X]
        n12 = n1 // 2
        n11 = n1 - n12
        n22 = n2 // 2
        n21 = n2 - n22
        res = sim_fun(x1, y1, x2, y2, n11, n12, n21, n22)
        # Store results
        nnpvalue1[run, :] = res['nnpvalue1']
        llpvalue1[run, :] = res['llpvalue1']
        nnpvalue2[run] = res['nnpvalue2']
        llpvalue2[run] = res['llpvalue2']
        nnpvalue_gm[run, :] = res['nnpvalue.gm']
        llpvalue_gm[run, :] = res['llpvalue.gm']
        nnpvalue_hm[run, :] = res['nnpvalue.hm']
        llpvalue_hm[run, :] = res['llpvalue.hm']
        centropy_ll_joint[run] = res['centropy.ll.joint']
        centropy_ll_marginal[run] = res['centropy.ll.marginal']
        centropy_nn_joint[run] = res['centropy.nn.joint']
        centropy_nn_marginal[run] = res['centropy.nn.marginal']
        error_ll[run] = res['error.ll']
        error_nn[run] = res['error.nn']
        cerror12_ll_joint[run] = res['cerror12.ll.joint']
        cerror12_ll_marginal[run] = res['cerror12.ll.marginal']
        cerror12_nn_joint[run] = res['cerror12.nn.joint']
        cerror12_nn_marginal[run] = res['cerror12.nn.marginal']
        cerror22_ll_joint[run] = res['cerror22.ll.joint']
        cerror22_ll_marginal[run] = res['cerror22.ll.marginal']
        cerror22_nn_joint[run] = res['cerror22.nn.joint']
        cerror22_nn_marginal[run] = res['cerror22.nn.marginal']
    # Compute rejection rates
    llper_rej1 = np.mean(llpvalue1 < alpha, axis=0)
    nnper_rej1 = np.mean(nnpvalue1 < alpha, axis=0)
    llper_rej2 = np.mean(llpvalue2 < alpha)
    nnper_rej2 = np.mean(nnpvalue2 < alpha)
    llper_rej_gm = np.mean(llpvalue_gm < alpha, axis=0)
    nnper_rej_gm = np.mean(nnpvalue_gm < alpha, axis=0)
    llper_rej_hm = np.mean(llpvalue_hm < alpha, axis=0)
    nnper_rej_hm = np.mean(nnpvalue_hm < alpha, axis=0)
    # Combine all rejection rates into a single vector and take the mean
    all_rej = np.concatenate([
        nnper_rej1,          # length 3
        [nnper_rej2],        # length 1
        nnper_rej_gm,        # length 3
        nnper_rej_hm         # length 3
    ])
    power = np.mean(all_rej)
    return power















