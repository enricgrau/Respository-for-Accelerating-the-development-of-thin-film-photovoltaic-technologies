from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import spectrapepper as spep
import pandas as pd
import numpy as np

folder = '../data/models/'
target = 'voc'  # voc, jsc, eta, or ff

ra_0 = spep.load(folder+'/'+str(target)+'/ra_'+str(target)+'_areas_sens_analysis_new.txt', transpose=False)
pl_0 = spep.load(folder+'/'+str(target)+'/pl_'+str(target)+'_areas_sens_analysis_new.txt', transpose=False)

d1_tr, d2_tr, _, _, ta_tr = spep.load(folder+'/'+str(target)+'/d1d2_train_voc_(80;76;63;73;82;72;74).txt', transpose=True)
d1_te, d2_te, _, _, ta_te = spep.load(folder+'/'+str(target)+'/d1d2_test_voc_(80;76;63;73;82;72;74).txt', transpose=True)

train, test = 0, 0
for i in range(1):
    ra, pl = spep.shuffle([ra_0, pl_0])
    ra, pl = np.transpose(ra), np.transpose(pl)
    
    div = int(0.7*len(ra[0]))

    ra0_tr, ra1_tr, ra2_tr, ra3_tr = ra[0][:div], ra[1][:div], ra[2][:div], ra[3][:div]
    pl0_tr = pl[0][:div]
    
    ra0_te, ra1_te, ra2_te, ra3_te = ra[0][div:], ra[1][div:], ra[2][div:], ra[3][div:]
    pl0_te = pl[0][div:]
    
    x_tr = spep.mergedata([ra0_tr, ra1_tr, ra3_tr, pl0_tr])
    x_te = spep.mergedata([ra0_te, ra1_te, ra3_te, pl0_te])
    y_tr = ra[4][:div]
    y_te = ra[4][div:]

    poly_model = PolynomialFeatures(degree=3, interaction_only=False)
    poly_x_val_tr = poly_model.fit_transform(x_tr)
    poly_x_val_te = poly_model.transform(x_te)
    
    poly_model.fit(poly_x_val_tr, y_tr)
    
    regression_model = LinearRegression()
    regression_model.fit(poly_x_val_tr, y_tr)
    
    y_pred_tr = regression_model.predict(poly_x_val_tr)
    y_pred_te = regression_model.predict(poly_x_val_te)
    
    coef = regression_model.coef_
    
    r2_tr = r2_score(y_tr, y_pred_tr)
    r2_te = r2_score(y_te, y_pred_te)

    print(i, r2_tr, r2_te, 0.7*r2_tr+0.3*r2_te)

    for i,j in zip(y_tr, y_pred_tr):
        plt.scatter(i, j, c='black', s=1)
    for i,j in zip(y_te, y_pred_te):
        plt.scatter(i, j, c='blue', s=1)
    plt.ylim(0,500)
    plt.xlim(0,500)
    plt.show()
