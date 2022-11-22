import spectrapepper as spep

ra = spep.load('../data/_ra_raw.txt')
x_ra = [i[0] for i in spep.load('../data/x_raman.txt')]

ra = spep.bspbaseline(ra, x_ra, [117, 347, 445, 539] , avg=10, plot=False, remove=True)
ra = spep.normtoratio(ra, [173, 179], [120, 271], x=x_ra)
ra = spep.normtoglobalmax(ra)

pl = spep.load('../data/_pl_raw.txt')
x_pl = spep.load('../data/x_pl.txt')[0]

pl = spep.normtoglobalmax(pl)

rapl = spep.mergedata([ra, pl])
