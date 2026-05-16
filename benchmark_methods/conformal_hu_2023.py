#conformal testing benchmark:
import numpy as np
from scipy.stats import f, norm
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

def NNfun(X, Z, xpre1, xpre2, nnrep = 10, hidden_layers = [8, 3],
    acfun = 'sigmoid', optim = 'SGD', n_epoch = 200, learning_rate = 1e-3, l1 = 0):
    n1 = xpre1.shape[0]
    n2 = xpre2.shape[0]
    prob1 = np.zeros((n1, nnrep))
    prob2 = np.zeros((n2, nnrep))
    for i in range(nnrep):
        fit_nn = 


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


def KLR_cv(X, Y, xpre, ypre, lambdaseq, sigmaseq):
    lambda_len = len(lambdaseq)
    sigma_len = len(sigmaseq)
    centropy = np.zeros((lambda_len, sigma_len))
    ypre = ypre.astype(int)
    minerror = np.inf
    for i, lambda_now in range(lambda_len):
        for j, sigma in range(sigma_len):
            cv_data = kFold(X, Y, shuffle = True)
            pipe = Pipeline([
                ('rbf': RBFSampler()),
                ('lr': LogisticRegression())])
            param_grid = {
            'rbf__gamma': sigmaseq,
            'lr__C': 1/np.array(lambdaseq)
            }
            fit_klr = 
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
    
                
             





















