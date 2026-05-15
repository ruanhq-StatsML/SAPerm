#conformal testing benchmark:

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

def getEcdf2(v):
    v_sorted = np.sort(v)
    ecdf1 = np.searchsorted(v_sorted, v, side = 'right')/len(v)
    ecdf2 = np.array([np.mean(v == x) for x in v])
    return ecdf1, ecdf2

def getEcdf1(v1, v2):
    v2_sorted = np.sort(v2)
    point1 = np.searchsorted(v2_sorted, v1, side = 'right')/len(v2)
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









def NNfun(x, z, xpre1, xpre2, nnrep = 10):
    n1 = x.shape[0]
    n2 = z.shape[0]
    prob1 = np.zeros((n1, nnrep))
    prob2 = np.zeros((n2, nnrep))
    for i in range(nnrep):
        fit_nn = MLPClassifier(
            hidden_layer_sizes = hidden_layers,
            activation = '',
            solver = ,
            alpha = 0.0, 
            learning_rate = 1e-5,

        	)























