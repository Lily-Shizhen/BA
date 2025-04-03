import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

##############################################################################
# 1) Define parameters
##############################################################################

# Tested dimensions (rows of the heatmap)
d_list = [16, 32, 64, 128, 256]

# Full list of k-values in the preprocessing (columns before selection)
demon_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Selected k-values to display in the heatmap (must be in demon_list)
selected_ks = [1, 4, 16, 64, 256, 1024, 4096, 16384]

# Noise settings (keys) and their corresponding pickle file paths.
# Adjust these paths as needed (relative to your content root or absolute).
noise_settings = [1, 10, 20, 40]
results_files = {
    1: '/Users/shizhenli/Documents/bachelor thesis/Code/basis_factor_1/results.pickle',
    10: '/Users/shizhenli/Documents/bachelor thesis/Code/basis_factor_10/results.pickle',
    20: '/Users/shizhenli/Documents/bachelor thesis/Code/basis_factor_20/results.pickle',
    40: '/Users/shizhenli/Documents/bachelor thesis/Code/basis_factor_40/results.pickle'
}

##############################################################################
# 2) Create a 2x2 subplot layout for the 4 noise settings
##############################################################################
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))


##############################################################################
# 3) For each noise setting, load the corresponding results and build a heatmap
##############################################################################
# For the x-axis of the heatmap, we use the selected k-values.
num_selected = len(selected_ks)

for ax, noise in zip(axs.flatten(), noise_settings):
    # Load the results file for the given noise setting
    with open(results_files[noise], 'rb') as f:
        results = pickle.load(f)

    # Create an empty risk matrix of shape (num_dimensions, num_selected_ks)
    risk_matrix = np.zeros((len(d_list), num_selected))

    # Fill the risk matrix: each row corresponds to a tested dimension (from d_list)
    # and each column corresponds to a selected k-value.
    for i, d in enumerate(d_list):
        for j, k_val in enumerate(selected_ks):
            if k_val in demon_list:
                idx = demon_list.index(k_val)
                # Here we assume results[d]['l_loss'] is an array whose length equals len(demon_list)
                risk_matrix[i, j] = results[d]['l_loss'][idx]
            else:
                risk_matrix[i, j] = np.nan

    # Plot the heatmap for this noise setting
    im = ax.imshow(risk_matrix, aspect='auto', cmap='viridis', origin='lower')
    ax.set_title(f'Noise factor: {noise}', fontsize=14)
    ax.set_xlabel('k (demonstration length)', fontsize=12)
    ax.set_ylabel('Dimension (d)', fontsize=12)

    # Set x-ticks: one tick per selected k-value
    ax.set_xticks(np.arange(num_selected))
    ax.set_xticklabels(selected_ks, rotation=45, fontsize=10)

    # Set y-ticks: one tick per tested dimension
    ax.set_yticks(np.arange(len(d_list)))
    ax.set_yticklabels(d_list, fontsize=10)

##############################################################################
# 4) Add a common colorbar
##############################################################################
#fig.colorbar(im, ax=axs.ravel().tolist(), shrink=0.6, label='ICL Risk')
# Increase horizontal/vertical spacing among subplots
plt.subplots_adjust(left=0.08, right=0.88, top=0.92, bottom=0.08,
                    wspace=0.3, hspace=0.3)

# Now create the colorbar with a bit of shrink or fraction/pad
cbar = fig.colorbar(im, ax=axs.ravel().tolist(),
                    shrink=0.8,    # shrink colorbar a bit
                    pad=0.02)     # put a little space between the bar and plots
#plt.tight_layout()
#plt.subplots_adjust(hspace=0.4)  # or 0.5, 0.6, etc.
plt.savefig('Heatmap_Noise_vs_Dim_vs_k(basis).pdf', bbox_inches='tight')
plt.show()