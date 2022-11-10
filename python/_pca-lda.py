from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import my_functions as spep
import pandas as pd
import numpy as np
import math
import time

start = time.time()

folder = '../data/'

opto = spep.load(folder+'new_op_x.txt')
plraman = spep.load(folder+'pl(glomax)_ra(ratio+glomax).txt', fromline=0)

nombres = ['JSC', 'VOC', 'FFF', 'ETA']
limits = [[21.1, 27.3, 30.2], [320.0, 402.0, 429.0], [37, 51, 60], [2.5, 5, 7.5]] #[20, 25, 30], [300, 360, 400] ([330, 390, 430]), [35, 45, 60], [3, 5, 7]
optopos = [3, 4, 5, 6]
pca_dims = [350, 250, 350, 250]

print('+-----+------+------+------+------+------+------+------+------+------+')
print('| TAR | TR0  | TR1  | TR2  | TR3  | TE0  | TE1  | TE2  | TE3  | CV5  |')
print('+-----+------+------+------+------+------+------+------+------+------+')

for k in range(4):
    targets = [opto[l][optopos[k]] for l in range(len(opto))]
    class_targets, group_names = spep.classify(targets, gnumber=0, glimits=limits[k])
    
    features, class_targets, targets = spep.shuffle([plraman, class_targets, targets], delratio=0)

    ratio = 0.70
    div = int(math.floor(ratio*len(features)))

    features_train = pd.DataFrame(features[:div])
    features_test = pd.DataFrame(features[div:])
    targets_train = class_targets[:div]
    targets_test = class_targets[div:]

    pca_components = pca_dims[k]
    pca = PCA(n_components=pca_components).fit(features_train)
    pca_train = pca.transform(features_train)
    pca_test = pca.transform(features_test)

    lda = LinearDiscriminantAnalysis(n_components=2, solver='svd')
    PCs = lda.fit(pca_train, targets_train).transform(pca_train)
    pcalda_train_pred = lda.predict(pca_train)
    PCs2 = lda.transform(pca_test)
    pcalda_test_pred = lda.predict(pca_test)

    cm_tr = spep.confusionmatrix(targets_train, pcalda_train_pred, plot=False)
    cm_tr = [[round(j, 2) for j in i] for i in cm_tr]
    
    cm_te = spep.confusionmatrix(targets_test, pcalda_test_pred, plot=False)
    cm_te = [[round(j, 2) for j in i] for i in cm_te]

    crossval = cross_validate(lda, pca_test, targets_test, cv=5)['test_score']
    cv = round(np.mean(crossval), 2)
    
    df1 = pd.DataFrame(data=PCs, columns=['D1', 'D2'])
    df2 = pd.DataFrame(data=targets_train, columns=['T'])
    train_df = pd.concat([df1, df2], axis=1)

    df12 = pd.DataFrame(data=PCs2, columns=['D1', 'D2'])
    df22 = pd.DataFrame(data=targets_test, columns=['T'])
    test_df = pd.concat([df12, df22], axis=1)

    # spep.plot2dml(train_df, test_df, labels=['C1', 'C2', 'C3', 'C4'], xax='D1', yax='D2') # uncomment to plot D1 vs D2

    print(f'| {nombres[k]} | {cm_te[0][0]} | {cm_te[0][0]} | {cm_te[0][0]} | {cm_te[0][0]} | {cm_te[0][0]} | {cm_te[1][1]} | {cm_te[2][2]} | {cm_te[3][3]} | {cv} |')
    print('+-----+------+------+------+------+------+------+------+------+------+')    

print(round(time.time()-start, 0), ' s')
