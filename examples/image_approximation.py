
import os 

import numpy as np 
from scipy import datasets
import matplotlib.pyplot as plt

from blakezisserman import BlakeZisserman2D



######################################################################################################################
# DO NOT EDIT BELOW



# Set output directory and filenames
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "image")
summary_plot_fname = "summary.png"
os.makedirs(output_dir, exist_ok=True)



### Load the RGB test image from scipy (downloaded once via pooch)
g = datasets.face().astype(np.float32)



### Compute Blake-Zisserman image approximation for different parameter choices and store the results.



# Load image values into a Blake-Zisserman 2-dimensional model
bz2d = BlakeZisserman2D(values=g)



# Define a grid of parameters to evaluate the BZ model with different weights
mu    = 0.01 
alpha = 2. 
beta  = 1. 
delta = 30.
  
# Minimize BZ functional (omit breaks, convergence flag and functional output)
u, s, z, _, _ = bz2d.solve(mu, alpha, beta, delta, log_level='INFO')

g = g.astype(np.uint8)
u = np.clip(u, 0, 255).astype(np.uint8)   # u may slightly overshoot [0, 255]



### Plot the results.



# Prepare summary plot grid
print(f'Saving plots in: {output_dir}')

fig, axes = plt.subplots(2, 2, figsize=(12, 9.6), layout="constrained")

# 
axes[0, 0].imshow(g)
axes[0, 0].set_title("Original image")
axes[0, 0].axis('off')

axes[1, 0].imshow(u)
axes[1, 0].set_title("Approx image")
axes[1, 0].axis('off')

axes[0, 1].imshow(s, vmin=0, vmax=1)
axes[0, 1].set_title("Discontinuity")
axes[0, 1].axis('off')

axes[1, 1].imshow(z, vmin=0, vmax=1)
axes[1, 1].set_title("Gradient discontinuity")
axes[1, 1].axis('off')

# Save input data figure
fig.savefig(os.path.join(output_dir, summary_plot_fname), dpi=200, bbox_inches='tight')
plt.close(fig)

print('Done.')