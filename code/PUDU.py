import spectrapepper as spep
from pudu import pudu
from pudu import perturbation as ptn
from pudu import plots as plots
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# load data
X, y = spep.load('data/X_cis.txt'), spep.load('data/y_cis.txt')

y, _ = spep.classify(y, gnumber=0, glimits=[320, 402, 429])

# Load pre-trained LDA and PCA models
lda = pickle.load(open('data/lda_cis.sav', 'rb'))
pca = pickle.load(open('data/pca_cis.sav', 'rb'))

pred = lda.predict(pca.transform(X))

# is scaling, then apply inside `pf`
def pf(X):
    X = X[0,:,:,0]
    return lda.predict_proba(pca.transform(X))[0]

for j,k,l in zip(y, X, pred):
    if j == l and j < 3: #does not show top class. Only correct classifications

        x = k[np.newaxis, np.newaxis, :, np.newaxis]
        imp = pudu.pudu(x, j, pf)
        imp.importance(window=1, evolution=int(j+1), perturbation=ptn.Positive())
        
        plots.plot(imp.x, imp.imp)

        # to save the results you can save the following vector `vec`
        #vec = list(imp.imp[0, 0, :, 0])

