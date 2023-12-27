import numpy as np
import spectrapepper as spep
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate

X, y = spep.load('X_czgse.txt'), spep.load('y_czgse.txt')

X = StandardScaler().fit_transform(X)

y, _ = spep.classify(y, gnumber=0, glimits=[0.85, 1.05, 1.15])
# y, _ = spep.classify(y, gnumber=0, glimits=[690, 705])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)

pca_dims, lda_dims = 70, 2

pca = PCA(n_components=pca_dims).fit(X_train)
pca_train, pca_test = pca.transform(X_train), pca.transform(X_test)

lda = LinearDiscriminantAnalysis(n_components=lda_dims).fit(pca_train, y_train)
lda_train, lda_test = lda.transform(pca_train), lda.transform(pca_test)

cv = np.mean(cross_validate(lda, pca_test, y_test, cv=5)['test_score'])
train_score, test_score = lda.score(pca_train, y_train), lda.score(pca_test, y_test)

pred_train, pred_test = lda.predict(pca_train), lda.predict(pca_test)

cm_tr = spep.confusionmatrix(y_train, pred_train, plot=True)
cm_te = spep.confusionmatrix(y_test, pred_test, plot=True)
