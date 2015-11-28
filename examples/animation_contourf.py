import pylab as pl
import numpy as np
import matplotlib.animation as animation
import types

def setvisible(self,vis):
    for c in self.collections: c.set_visible(vis)
fig = pl.figure()
# Some 2D arrays to plot (time,x,y)
data = np.random.random_sample((20,10,10))

# ims is a list of lists, each row is a list of artists to draw in the
# current frame; here we are just animating one artist, the image, in
# each frame
ims = []
for i in range(len(data[:,0,0])):
    t_step = int(i)
    im = pl.contourf(data[i,:,:])

    im.set_visible = types.MethodType(setvisible,im)
    im.axes = pl.gca()
    im.figure=fig

    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=1000, blit=False,repeat_delay=1000)

pl.show()