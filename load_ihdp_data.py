def load_IHDP_data(training_data,testing_data,i=7):
    with open(training_data,'rb') as trf, open(testing_data,'rb') as tef:
        train_data=np.load(trf); test_data=np.load(tef)
        y=np.concatenate(   (train_data['yf'][:,i],   test_data['yf'][:,i])).astype('float32') #most GPUs only compute 32-bit floats
        t=np.concatenate(   (train_data['t'][:,i],    test_data['t'][:,i])).astype('float32')
        x=np.concatenate(   (train_data['x'][:,:,i],  test_data['x'][:,:,i]),axis=0).astype('float32')
        mu_0=np.concatenate((train_data['mu0'][:,i],  test_data['mu0'][:,i])).astype('float32')
        mu_1=np.concatenate((train_data['mu1'][:,i],  test_data['mu1'][:,i])).astype('float32')
        data={'x':x,'t':t,'y':y,'t':t,'mu_0':mu_0,'mu_1':mu_1}
        data['t']=data['t'].reshape(-1,1) #we're just padding one dimensional vectors with an additional dimension 
        data['y']=data['y'].reshape(-1,1)
        #rescaling y between 0 and 1 often makes training of DL regressors easier
        data['y_scaler'] = StandardScaler().fit(data['y'])
        data['ys'] = data['y_scaler'].transform(data['y'])
    return data

data=load_IHDP_data(training_data='./ihdp_npci_1-100.train.npz',testing_data='./ihdp_npci_1-100.test.npz')

data['t'] = 


#uplift curve:
from scipy.spatial.distance import cdist
def rbf_kernel(X, sigma = None):
    dists = cdist(X, X, 'sqeuclidean')
    if sigma is None:
        sigma = np.sqrt(np.median(dists))
        if sigma == 0:
            sigma = 1e-8
    K = np.exp(-dists/(2 * (sigma ** 2)))
    return K
def HSIC(X, Y):
    if X.ndim == 1:
        X = X[:, np.newaxis]
        Y = Y[:, np.newaxis]
    n = X.shape[0]
    K = rbf_kernel(X, sigma_X)
    L = rbf_kernel(Y, sigma_Y)
    #hat matrix:
    H = np.eye(n) - np.ones((n, n))/n
    Lc = H @ L @ H 
    Kc = H @ K @ H
    hsic_val = np.trace(Kc @ Lc)/((n-1) ** 2)
    return hsic_val











