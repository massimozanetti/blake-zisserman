
import os 

from itertools import product
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from blakezisserman import BlakeZisserman1D



######################################################################################################################
# Directory setup, signal construction and reproducibility.
######################################################################################################################

# Set piecewise differential signal construction parameters
length_1 = 4.0       # First segment span
slope_1 = -0.8       # First segment slope
length_2 = 3.0       # Second segment span
slope_2 = 0.5        # Second segment slope
radius = 2.0         # Circle radius 
amplitude = 1.0      # Sinusoid amplitude
wavelength = 5.0     # Sinusoid wavelength
periods = 1.0        # Sinusoid periods
unit_points = 10     # Number of discretization points per unit
noise_std = .1       # Gaussian noise standard deviation

# Set random generator for gaussian noise reproducibility
rng = np.random.default_rng(seed=21495)

######################################################################################################################
# DO NOT EDIT BELOW



# Set output directory and files
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "signal")
grid_plot_fname    = "gridplot.png"
summary_plot_fname = "summary.png"
runs_dir           = "runs"
os.makedirs(os.path.join(output_dir, runs_dir), exist_ok=True)



### Generate a piecewise differentiable signal including linear, circular and sinusoidal pieces, and add white gaussian noise to it.



# First straight segment
x1 = np.linspace(0, length_1, int(length_1*unit_points))
y1 = slope_1*x1
# Second straight segment
x2 = np.linspace(x1[-1], x1[-1] + length_2, int(length_1*unit_points))
y2 = y1[-1] + slope_2*(x2-x2[0])
# Semicircle: endpoints at the same height
y_join = y2[-1]
x_center = x2[-1] + radius
theta = np.linspace(np.pi, 0, int(2*radius*unit_points))
x_arc = x_center + radius*np.cos(theta)
y_arc = y_join - radius*np.sin(theta)
# Sinusoid
u = np.linspace(0, periods * wavelength, int(periods*wavelength*unit_points))
x_sine = x_arc[-1] + u
y_sine = y_join + amplitude * np.sin(2*np.pi*u/wavelength)
# Complete curve, without duplicate joining points
x = np.concatenate([x1, x2[1:], x_arc[1:], x_sine[1:]])
y = np.concatenate([y1, y2[1:], y_arc[1:], y_sine[1:]])
# Add gaussian white noise
noise = rng.normal(0, noise_std, len(x))
g = y + noise



### Compute Blake-Zisserman signal approximation for different parameter choices and store the results.



# Load signal values into a Blake-Zisserman 1-dimensional model
bz1d = BlakeZisserman1D(values=g)

# Define a grid of parameters to evaluate the BZ model with different weights
LAMBDA = np.logspace(-1, -6, num=6)
ETA    = np.logspace(-2, -7, num=6)

# Results array allocation
shape_ = (len(LAMBDA), len(ETA), len(g))
u = np.empty(shape_)
z = np.empty(shape_)


# Minimize BZ functional (omit breaks, convergence flag and functional output)
for (i, llambda), (j, eta) in product(enumerate(LAMBDA), enumerate(ETA)):
  
    # Minimize BZ functional (omit breaks, convergence flag and functional output)
    u_, z_, _, _, _ = bz1d.solve(llambda, eta, bound_cond='periodic', log_level='INFO')

    u[i, j, :] = u_.squeeze()
    z[i, j, :] = z_.squeeze()



### Plot the results.



# Prepare summary plot grid
print(f'Saving plots in: {output_dir}')

fig, axis = plt.subplots(len(LAMBDA), len(ETA), squeeze=False, layout='constrained', figsize=(18, 13)) 

# Populate plots
for (i, llambda), (j, eta) in product(enumerate(LAMBDA), enumerate(ETA)):
  
    # Populate the summary grid plot
    axis[i, j].plot(x, g      , '.k', ms=1)
    axis[i, j].plot(x, u[i, j], '-r', lw=1)
    axis[i, j].set_xticks([])
    axis[i, j].set_yticks([])
    axis[i, j].set_title(f'λ={llambda:.0e}, η={eta:.0e}', fontsize=16)
    
    # Populate individual plot
    fig_= plt.figure(figsize=(18,15))
    gs = gridspec.GridSpec(2, 2, figure=fig_, hspace=0, wspace=0.15)
    ax0 = fig_.add_subplot(gs[:, 0])
    ax1 = fig_.add_subplot(gs[0, 1])
    ax2 = fig_.add_subplot(gs[1, 1])

    ax0.plot(x, g      , '.k', ms=1, label='g')
    ax0.plot(x, u[i, j], '-r', lw=1, label='u')
    ax0.set(xlabel="x", ylabel="y")
    ax0.set_aspect(1, adjustable="box")
    ax0.grid(alpha=0.3)
    ax0.legend(loc='upper right')

    ax1.plot(x, g, '.k', ms=1, label='g')
    ax1.plot(x, u[i, j], '-r', ms=1, label='u')
    ax1.tick_params(labelbottom=False)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right')

    ax2.plot(x, z[i, j], '-g', ms=1, label='z')
    ax2.set_ylim(-0.1, 1.1)
    ax2.set_yticks([0.0, 0.5, 1.0])
    ax2.set(xlabel="t")
    ax2.grid(alpha=0.3)
    ax2.legend(loc='upper right')

    fig_.suptitle(f'λ={llambda:.0e}, η={eta:.0e}', fontsize=16)

    # Save individual plot figure
    filename = f"lambda_{llambda:.0e}_eta_{eta:.0e}.png".replace('+', '')
    fig_.savefig(os.path.join(output_dir, runs_dir, filename), dpi=200, bbox_inches='tight')
    plt.close(fig_)

# Save grid plot figure
fig.savefig(os.path.join(output_dir, grid_plot_fname))
plt.close(fig)

# Input and best approx data plot
i_, j_ = 1, 2
llambda = LAMBDA[i_]
eta     = ETA[j_]

fig, ax = plt.subplots()
ax.plot(x, y, '-b', lw=1, label=r'Ideal signal: $y$')
ax.plot(x, g, '.k', ms=1, label=r'Noisy signal: $g = y + \mathcal{N}(0, \sigma)$')
ax.plot(x, u[i_, j_], '--r', lw=1, label=f'Best approx: λ={llambda:.0e}, η={eta:.0e}')
ax.set(xlabel="x", ylabel="y")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(alpha=0.3)
ax.legend(loc='upper right')
ax.set_title("Approximation of piecewise differentiable signal")
# Save summary plot figure
fig.savefig(os.path.join(output_dir, summary_plot_fname), dpi=200, bbox_inches='tight')
plt.close(fig)

print('Done.')