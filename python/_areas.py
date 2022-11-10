import my_functions as spep

y = spep.load('../data/pl(glomax)_ra(ratio+glomax).txt')

ra = [i[:1024] for i in y]
pl = [i[1024:] for i in y]

x_ra = [i[0] for i in spep.load('../data/x_raman.txt')]
x_pl = spep.load('../data/x_pl.txt')[0]

limits_ra = [[145, 160], [169, 178], [206, 215], [227, 240]] 
areas_ra = spep.areacalculator(ra, x_ra, limits_ra)

limits_pl = [1200, 1400] 
areas_pl = spep.areacalculator(pl, x_pl, limits_pl)
