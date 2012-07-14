from matplotlib.pylab import *
from FFABench_cy import FFABench
rep = FFABench()
fig,axL = subplots(nrows=2)
axL[1].plot(rep['P'],rep['s2n'],label='Periodogram')
sca(axL[1])
xlabel('Period')
ylabel('S/N')
legend()
draw()
show()
