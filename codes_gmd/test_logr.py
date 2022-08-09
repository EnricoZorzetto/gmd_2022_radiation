

import numpy as np
import  matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import pandas as pd

# X, y = load_iris(return_X_y=True)
# y[y>1]=1 # binary problem
# clf = LogisticRegression(random_state=0, solver="newton-cg").fit(X, y)
# PRED = clf.predict(X)
# PROB = clf.predict_proba(X)
# # PRED = clf.predict(X[:10, :])
# # PROB = clf.predict_proba(X[:10, :])
# clf.score(X, y)
#
# PROB.shape
#
# plt.figure()
# plt.plot(PROB[:,1], 'o')
# plt.plot(y, '*k')
# plt.show()

# X.shape
# X[:2, :].shape
#
# PRED.shape
# PROB
#
# y

N = 1000
x1 = np.random.randn(N)
x2 = np.random.randn(N)
x3 = np.random.randn(N)

# df = pd.DataFrame({'A':x1, 'B':x2, 'C':x3})
df = pd.DataFrame({'A':x1})

thresh = -1
a1 = 2.0; a2 = 0.0; a3=0.0; a4=0
# ytrue = pd.Series( a1*x1 + a2*x2 + a3*x3 + a4 )
ytrue = pd.Series( a1*x1 )
ycens = ytrue.copy()
left = ytrue < thresh
ytrue = ytrue.clip(lower=thresh)
# ycens[ytrue < thresh] = thresh
cens = pd.Series(np.zeros(N))
cens[left] = -1

prob_above = 1-(np.sum(left))/np.size(left)

plt.figure()
plt.plot(x1, ytrue, 'ob')
plt.plot(x1, ycens, 'or')
plt.show()

from tobit import TobitModel
tr = TobitModel()
lr = LinearRegression()
result = tr.fit(df, ytrue, cens, verbose=False)

result_lr = lr.fit(df, ytrue)

ypred = tr.predict(df)
ypred_lr = lr.predict(df)

print( tr.coef_ )
print( tr.ols_coef_ )
print( lr.coef_ )
print( tr.intercept_ )
print( lr.intercept_ )

plt.figure()
plt.plot(ytrue, ypred*prob_above, 'sr')
plt.plot(ytrue, ypred_lr, 'og')
plt.plot(ytrue, ytrue, '-k')
plt.show()


