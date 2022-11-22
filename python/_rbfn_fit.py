from localreg import RBFnet, plot_corr
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import spectrapepper as spep
import numpy as np

folder = '../data/models/'

opt = 'eta' # voc, jsc, eta, or ff

ra = spep.load(folder+opt+'/ra_'+str(opt)+'_areas_sens_analysis_new.txt', transpose=True)
pl = spep.load(folder+opt+'/pl_'+str(opt)+'_areas_sens_analysis_new.txt', transpose=True)
target = ra[4]
ra0, ra1, ra2, ra3 = ra[0], ra[1], ra[2], ra[3]
pl0 = pl[0]
target, ra0, ra1, ra2, ra3, pl0 = spep.shuffle([target, ra0, ra1, ra2, ra3, pl0])

train, test = 0, 0

target, ra0, ra1, ra2, ra3, pl0 = spep.shuffle([target, ra0, ra1, ra2, ra3, pl0])

target_te = target[1494:]
ra0_te, ra1_te, ra2_te, ra3_te = ra0[1494:], ra1[1494:], ra2[1494:], ra3[1494:]
pl0_te  = pl0[1494:]

target_tr = target[:1494]
ra0_tr, ra1_tr, ra2_tr, ra3_tr = ra0[:1494], ra1[:1494], ra2[:1494], ra3[:1494]
pl0_tr = pl0[:1494]

input = np.array([ra0_tr, ra1_tr, ra2_tr, ra3_tr, pl0_tr]).T
z = target_tr

net = RBFnet()
net.train(input, z, num=200, verbose=False)
z_hat = net.predict(input)
z_hat_te = net.predict(np.array([ra0_te, ra1_te, ra2_te, ra3_te, pl0_te]).T)

test_r2 = round(r2_score(target_te, z_hat_te))
train_r2 = round(r2_score(target_tr, z_hat))

print(train_r2, test_r2)
