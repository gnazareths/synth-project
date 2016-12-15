import numpy as np
from scipy.optimize import fmin_slsqp

def w_rss(w, v, x0, x1):
    k = len(x1)
    importance = np.zeros((k,k))
    np.fill_diagonal(importance, v)
    predictions = np.dot(x0, w)
    errors = x1 - predictions
    weighted_errors = np.dot(errors.transpose(), importance)
    weighted_rss = np.dot(weighted_errors,errors).item(0)
    return weighted_rss
    
def v_rss(w, z0, z1):
    predictions = np.dot(z0,w)
    errors = z1 - predictions
    rss = sum(errors**2)
    return rss

def w_constraint(w, v, x0, x1):
    return np.sum(w) - 1
    
def v_constraint(w, v, x0, x1, z0, z1):
    return np.sum(v) - 1
    
def get_w(w, v, x0, x1):
    result = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)
    weights = result[0]
    print weights
    return weights

def get_v_0(v, w, x0, x1, z0, z1):
    weights = fmin_slsqp(w_rss, w, f_eqcons=w_constraint, bounds=[(0.0, 1.0)]*len(w),
             args=(v, x0, x1), disp=False, full_output=True)[0]
    print weights
    rss = v_rss(weights, z0, z1)
    return rss
    
def get_v_1(v, w, x0, x1, z0, z1): ### This function is not optimizing anything. It's returning the initial value of v. :(
    result = fmin_slsqp(get_v_0, v, f_eqcons=v_constraint, bounds=[(0.0, 1.0)]*len(v),
             args=(w, x0, x1, z0, z1), disp=False, full_output=True)
    importance = result[0]
    print result
    print '---'
    print importance
    return importance
    
## these are for testing:
    
X0 = np.array([[3,4,5,3],
               [1,2,2,4],
               [6,5,3,7],
               [1,0,5,2]])  

X1 = np.array([4, 2, 4, 2]).transpose() 

Z0 = np.array([[11,8,20,14],
               [10,8,21,14],
               [12,10,25,17],
               [13,11,27,20],
               [13,11,27,21]])

Z1 = np.array([16, 17, 22, 25, 25])

V = np.array([.25,.25,.25,.25])
V1 = np.zeros((4,4))
np.fill_diagonal(V1,V)

W = np.array([.5, .0, .25, .25]).transpose()
    
get_w(W,V,X0,X1)
get_v_0(V,W,X0,X1,Z0,Z1)
get_v_1(V,W,X0,X1,Z0,Z1)


print "First RSS, with all weights and predictors equal:"
print v_rss(W,Z0,Z1)

print "Second RSS, with all weights set, but predictors equal:"
weights = get_w(W,V,X0,X1)
print v_rss(weights, Z0, Z1)

print "Third RSS, with weights and predictors:"
imp = get_v_1(V,W,X0,X1,Z0,Z1)
imp_w = get_w(W,imp,X0,X1)
print v_rss(imp_w,Z0,Z1)
