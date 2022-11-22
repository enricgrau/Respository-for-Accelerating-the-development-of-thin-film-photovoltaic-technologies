import matplotlib.pyplot as plt
import spectrapepper as spep

folder = '../data/'

plra = spep.load(folder+'_pl_raman_processed_merged.txt')
ra, pl = [i[:1024] for i in plra], [i[1024:] for i in plra]
plra = [ra, pl]

_, _, _, jsc, voc, fff, eta = spep.load(folder+'_optoelectronics.txt', transpose=True)
optos, opto_n = [jsc, voc, fff, eta], ['jsc', 'voc', 'fff', 'eta']

limits = [[21.1, 27.3, 30.2], [251, 404, 426], [37, 51, 60], [2.5, 5, 7.5]]
spec = ['ra', 'pl']

for j in range(len(limits)):
    group, _ = spep.classify(optos[j], gnumber=0, glimits=limits[j])
    for l in range(max(group)+1):        
        for i in range(len(plra)):
            temp = []
            for k in range(len(plra[i])):
                if group[k] == l:
                    temp.append(plra[i][k])
            
            avg = spep.avg(temp)
            typ = spep.typical(temp)      
            rep = spep.representative(temp)
            sdv = spep.sdev(temp)  
            minn, maxx = spep.minmax(temp)
            upp = avg+sdv
            low = avg-sdv
            
            plt.title(f'{spec[i]}, {opto_n[j]}, class {l}')
            plt.plot(avg, lw=2, c='red', label='Avg.')
            plt.plot(typ, lw=2, c='orange', label='Typ.')
            plt.plot(rep, lw=2, c='blue', label='Rep.')
            plt.xlim(100, 400)
            plt.ylim(0, 0.1)
            plt.legend()
            plt.show()
