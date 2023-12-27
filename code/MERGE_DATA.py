import spectrapepper as spep

ra = spep.load('data/raman_cis.txt')
pl = spep.load('data/pl_cis.txt')

x_ra = spep.load('data/axis_raman.txt')

ra = spep.normtopeak(ra, x=x_ra, peak=152, shift=2)
ra = spep.normtoglobalmax(ra)

pl = spep.normtoglobalmax(pl)

# here we jsut merge the first 100 because this may take a while
# to merge all the data remove `[:100]` from the line below
rapl = spep.mergedata([ra[:100], pl[:100]]) 

# plot
import matplotlib.pyplot as plt
for i in rapl:
    plt.plot(i)
plt.show()

#np.savetxt('data/SOME_NAME.txt', rapl, fmt="%s")

