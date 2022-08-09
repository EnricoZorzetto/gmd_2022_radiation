
import numpy as np

# check transform between X and Y
# X = np.array([ 1.0, 1.0, 3.0, 3.0, 5.0, 5.0, 5.0])
# X = np.array([ 1.0, 1.0, 1.0, 1.05])


X = np.array([-0.8, -0.1])
# W = np.array([0.01, 0.99])
W = np.array([0.99, 0.01])


# Ybar = -10.0 # imposed new average value
# lowerbound = -1.0 # imposed variable lower bound

def rescale_preserving_minval(X, Ybar=1.0, lowerbound = -1.0, weights=None):

    Xmin = np.min(X)
    if weights is not None:
        Xbar = np.sum(weights * X)
    else:
        Xbar = np.mean(X)
    Ymin = max(Ybar - (Xbar - Xmin), lowerbound)
    if (Xbar <= Xmin) or (Ybar <= Ymin):
        Y = np.ones(np.shape(X))*Xbar
    else:
        Y = Ybar + (Ybar-Ymin)*(X-Xbar)/(Xbar-Xmin)
    return Y

Y = rescale_preserving_minval(X, Ybar=-0.99, lowerbound=-1, weights=W)

# xlowerbound = -1.0  # for direct flux (=complete shade)
# xnewmean = aveterrain_fdir
# xnewmin = max(xlowerbound, xnewmean - (xoldmean - xoldmin))



print('X={}'.format(X))
print('Y={}'.format(Y))
print("Ybar = {}, Xbar = {}".format(np.sum(Y*W), np.sum(X*W)))
# print("Ystdv = {}, Xstdv = {}".format(np.sum((Y-)*W), np.stsum)(Y-)W)
print("Ymin = {}; Xmin = {}".format(np.min(Y), np.min(X)))
print("Ymax = {}; Xmax = {}".format(np.max(Y), np.max(X)))

