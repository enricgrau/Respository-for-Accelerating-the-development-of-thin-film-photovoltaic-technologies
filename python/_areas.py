import spectrapepper as spep

folder = '../data/'

y = spep.load(folder+'_pl_raman_processed_merged.txt')

ra = [i[:1024] for i in y]
pl = [i[1024:] for i in y]

x_ra = [i[0] for i in spep.load(folder+'_x_axis_raman.txt')]
x_pl = spep.load(folder+'_x_axis_pl.txt')[0]

limits_ra = [[145, 160], [169, 178], [206, 215], [227, 240]] 
areas_ra = spep.areacalculator(ra, x_ra, limits_ra)

limits_pl = [1200, 1400] 
areas_pl = spep.areacalculator(pl, x_pl, limits_pl)
