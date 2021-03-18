import numpy as np
import matplotlib.pylab as plt

"""
Plot the MountainCar loss surface and trajectories (glorot/handpicked 
initialization) on top. This generates Figure 1 in paper. 
"""
plt.figure(figsize=(6, 4))

# Plot background
arr = np.load("loss_surface.npz")
x, y, z = arr['x'], arr['y'], arr['z']
cs = plt.contourf(x, y, z.reshape(x.shape), 64)
plt.colorbar(cs)

# Plot trajectories
rand_chain = np.load("glorot_params.npz")['arr_0']
opt_chain = np.load("handpicked_params.npz")['arr_0']
plt.plot(rand_chain[:1300, 0], rand_chain[:1300, 1], 'w', linewidth=0.7)
plt.plot(opt_chain[:, 0], opt_chain[:, 1], c='xkcd:grey', linewidth=0.7)
plt.plot(rand_chain[0, 0], rand_chain[0, 1], 'm+')
plt.plot(opt_chain[0, 0], opt_chain[0, 1], 'm+')

# Fig stuff
plt.xlabel("policy parameter 0")
plt.ylabel("policy parameter 1")
plt.tight_layout(rect=(0, 0.01, 1, 1))
plt.savefig("toto.png", dpi=300)
plt.show()
