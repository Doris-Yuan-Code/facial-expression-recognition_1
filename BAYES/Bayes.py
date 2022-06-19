import numpy as np
import pandas as pd
from DataProcess import combineSeqData, combineLBPSeqData
from sklearn import model_selection
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.decomposition import PCA

Data = pd.read_csv('pdData.csv')
X0 = combineSeqData(Data)
y0 = np.array(Data['Face'].values)

# PCA方法
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X0)
X1 = pca.transform(X0)

# LBP方法
X2 = combineLBPSeqData(Data)

# LBP + PCA方法
pca = PCA(n_components=27, svd_solver='auto',
          whiten=True).fit(X2)
X3 = pca.transform(X2)

num_folds = 10
scoring = 'accuracy'

print("Naive Bayes：")
print("The face label of the dataset contains:", list(set(Data['Face'])))
for name, data in (["Raw", X0], ["PCA", X1], ["LBP", X2], ["LBP + PCA", X3]):
    kfold = model_selection.KFold(n_splits=num_folds)
    cv_results = model_selection.cross_val_score(GaussianNB(), data, y0, cv=kfold, scoring=scoring)
    msg = "%s NB: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


