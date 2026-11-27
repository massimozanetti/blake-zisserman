
import os 

from itertools import product
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from blakezisserman import BlakeZisserman1D



######################################################################################################################
# Directory setup, curve construction and reproducibility.
######################################################################################################################

# Set U-shaped polygon construction parameters
L = 10.              # Long side length
S = 3.               # Short side length
alpha = 30           # Rotation angle, clockwise (degrees)
unit_points = 5      # Number of discretization points per unit
noise_std = .1       # Gaussian noise standard deviation

# Set random generator for gaussian noise reproducibility
rng = np.random.default_rng(seed=69995)

######################################################################################################################
# DO NOT EDIT BELOW



# Set output directory and files
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "polygon")
grid_plot_fname    = "gridplot.png"
summary_plot_fname = "summary.png"
runs_dir           = "runs"
os.makedirs(os.path.join(output_dir, runs_dir), exist_ok=True)



### Generate the U-shaped polygon and add white gaussian noise to it.



# Define primary structural vertices sequentially (clockwise)
vertices = np.array([
    [0.0, 0.0],
    [0.0, L  ],  # Top-left outer tip
    [S  , L  ],  # Top-left inner corner
    [S  , S  ],  # Bottom-left interior base
    [L-S, S  ],  # Bottom-right interior base
    [L-S, L  ],  # Top-right inner corner
    [L  , L  ],  # Top-right outer tip
    [L  , 0.0],  # Bottom-right outer base
    [0.0, 0.0]   # Bottom-left outer base
])
xy = []
n = len(vertices)-1
# Interpolate points between vertices
for i in range(n):
    pt1 = vertices[i]
    pt2 = vertices[i+1]
    # Linearly interpolate between the two structural vertices
    points = np.linspace(pt1, pt2, num=int(np.linalg.norm(pt1-pt2))*unit_points, endpoint=False if i<n-1 else True)
    xy.append(points)
# Stack points into a 2d array
xy = np.vstack(xy)
# Construct 2d rotation matrix
theta = np.radians(alpha)
c, s = np.cos(theta), np.sin(theta)
R = np.array(((c, -s), (s, c)))
# Rotate the points and keep coordinates positive
xy = ((xy-np.array([L, 0])) @ R) + np.array([L, 1])
# Add gaussian white noise
noise = rng.normal(0, noise_std, xy.shape)
g = xy + noise



### Compute Blake-Zisserman curve approximation for different parameter choices and store the results.



# Load curve values into a Blake-Zisserman 1-dimensional model
bz1d = BlakeZisserman1D(values=g)

# Define a grid of parameters to evaluate the BZ model with different weights
LAMBDA = np.logspace(-2, -7, num=6)
ETA    = np.logspace(-2, -7, num=6)

# Results array allocation
u = np.empty((len(LAMBDA), len(ETA), g.shape[0], g.shape[1]))
z = np.empty((len(LAMBDA), len(ETA), g.shape[0]))

# Minimize BZ functional (omit breaks, convergence flag and functional output)
for (i, llambda), (j, eta) in product(enumerate(LAMBDA), enumerate(ETA)):

    # Minimize BZ functional (omit convergence flag and functional output)
    u_, z_, _, _, _ = bz1d.solve(llambda, eta, bound_cond='periodic', log_level='INFO')

    u[i, j, :, :] = u_
    z[i, j, :] = z_.squeeze()



### Plot the results.



# Prepare summary plot grid
print(f'Saving plots in: {output_dir}')

fig, axis = plt.subplots(len(LAMBDA), len(ETA), squeeze=False, layout='constrained', figsize=(18, 13)) 

# Poulate plots
for (i, llambda), (j, eta) in product(enumerate(LAMBDA), enumerate(ETA)):
    
    # Populate the summary grid plot
    axis[i, j].plot(g[:, 0], g[:, 1], '.k', ms=1)
    axis[i, j].plot(u[i, j, :, 0], u[i, j, :, 1], '-r', lw=1)
    axis[i, j].set_aspect("equal", adjustable="datalim")
    axis[i, j].set_xticks([])
    axis[i, j].set_yticks([])
    axis[i, j].set_title(f'λ={llambda:.0e}, η={eta:.0e}', fontsize=16)
    
    # Populate individual plot
    fig_= plt.figure(figsize=(18,15))
    gs = gridspec.GridSpec(3, 2, figure=fig_, hspace=0, wspace=0.15)
    ax0 = fig_.add_subplot(gs[:, 0])
    ax1 = fig_.add_subplot(gs[0, 1])
    ax2 = fig_.add_subplot(gs[1, 1])
    ax3 = fig_.add_subplot(gs[2, 1]) 

    ax0.plot(g[:, 0], g[:, 1], '.k', ms=1, label='g')
    ax0.plot(u[i, j, :, 0], u[i, j, :, 1], '-r', lw=1, label='u')
    ax0.set(xlabel="x", ylabel="y")
    ax0.set_aspect(1, adjustable="box")
    ax0.grid(alpha=0.3)
    ax0.legend(loc='upper right')

    ax1.plot(g[:, 0], '.k', ms=1, label='g_x')
    ax1.plot(u[i, j, :, 0], '-r', ms=1, label='u_x')
    ax1.tick_params(labelbottom=False)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(g[:, 1], '.k', ms=1, label='g_y')
    ax2.plot(u[i, j, :, 1], '-r', ms=1, label='u_y')
    ax2.tick_params(labelbottom=False)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    ax3.plot(z[i, j], '-b', ms=1, label='z')
    ax3.set_ylim(-0.1, 1.1)
    ax3.set_yticks([0.0, 0.5, 1.0])
    ax3.set(xlabel="t")
    ax3.grid(alpha=0.3)
    ax3.legend(loc='upper right')

    fig_.suptitle(f'λ={llambda:.0e}, η={eta:.0e}', fontsize=16)

    # Save individual plot figure
    filename = f"lambda_{llambda:.0e}_eta_{eta:.0e}.png".replace('+', '')
    fig_.savefig(os.path.join(output_dir, runs_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig_)

# Save results grid plot figure
fig.savefig(os.path.join(output_dir, grid_plot_fname))
plt.close(fig)

# Input and best approx data plot
i_, j_ = 2, 3
llambda = LAMBDA[i_]
eta     = ETA[j_]

fig, ax = plt.subplots()
ax.plot(xy[:, 0], xy[:, 1], '-b', lw=1, label=r'Ideal polygon: $(x, y)$')
ax.plot(g[:, 0],   g[:, 1], '.k', ms=1, label=r'Noisy polygon: $g = (x, y) + \mathcal{N}(0, \sigma)$')
ax.plot(u[i_, j_, :, 0], u[i_, j_, :, 1], '--r', lw=1, label=f'Best approx: λ={llambda:.0e}, η={eta:.0e}')
ax.set(xlabel="x", ylabel="y")
ax.set_aspect("equal", adjustable="box")
ax.set_xlim(right=22.)
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
ax.set_title("Approximation of polygonal curve")
# Save summary plot figure
fig.savefig(os.path.join(output_dir, summary_plot_fname), dpi=200, bbox_inches='tight')
plt.close(fig)

print('Done.')