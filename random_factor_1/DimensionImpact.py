import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

##############################################################################
# 1) Choose your parameters & load results
##############################################################################

d_list = [16, 32, 64, 128, 256]  # Example dimension list
demon_list = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384]

# Let's pick these k-values:
selected_ks = [1, 4, 16, 64, 256, 1024, 4096, 16384]

save_folder = 'results/'
path = pathlib.Path(save_folder)
results_file = 'results.pickle'

##############################################################################
# 2) Load the data from results.pickle
##############################################################################
with open(results_file, 'rb') as handle:
    results = pickle.load(handle)

##############################################################################
# 3) Prepare data for plotting
##############################################################################

dimension_loss = {}
dimension_upper = {}

for k_val in selected_ks:
    if k_val not in demon_list:
        print(f"Warning: k={k_val} not found in demon_list.")
        continue
    idx = demon_list.index(k_val)
    losses_for_this_k = []
    uppers_for_this_k = []
    for d in d_list:
        all_loss = results[d]['l_loss']  # shape ~ (len(demon_list),)
        all_upper = results[d]['l_upper']
        losses_for_this_k.append(all_loss[idx])
        uppers_for_this_k.append(all_upper[idx])
    dimension_loss[k_val] = np.array(losses_for_this_k)
    dimension_upper[k_val] = np.array(uppers_for_this_k)

##############################################################################
# 4) Create subplots with 8 rows (one per selected k) and 2 columns (risk and bound)
##############################################################################

num_rows = len(selected_ks)  # Here, 8
fig, axs = plt.subplots(nrows=num_rows, ncols=2, figsize=(12, 4*num_rows), sharex=False, sharey=False)

# For color styling
color_loss = 'darkblue'
color_upper = 'cyan'

for i, k_val in enumerate(selected_ks):
    # Skip if data wasn't found
    if k_val not in dimension_loss:
        axs[i, 0].set_title(f'k={k_val} (not found)')
        axs[i, 1].set_title(f'k={k_val} (not found)')
        continue

    # Prepare data
    x_dim = np.array(d_list)
    y_loss  = dimension_loss[k_val]
    y_upper = dimension_upper[k_val]

    # --- Left subplot: ICL Risk ---
    ax_risk = axs[i, 0]
    ax_risk.plot(x_dim, y_loss, marker='o', linewidth=3, color=color_loss, label='ICL Loss')
    ax_risk.set_title(rf'$k={k_val}$', fontsize=16)
    if i == 0:
        ax_risk.set_ylabel('ICL Risk', fontsize=14)
    ax_risk.set_xlabel('Dimension', fontsize=12)
    ax_risk.tick_params(axis='both', labelsize=12)
    ax_risk.legend(fontsize=10)

    # --- Right subplot: Upper Bound ---
    ax_bound = axs[i, 1]
    ax_bound.plot(x_dim, y_upper, marker='s', linewidth=3, linestyle='--', color=color_upper, label='Upper Bound')
    if i == 0:
        ax_bound.set_ylabel('Upper Bound', fontsize=14)
    ax_bound.set_xlabel('Dimension', fontsize=12)
    ax_bound.tick_params(axis='both', labelsize=12)
    ax_bound.legend(fontsize=10)

plt.tight_layout()
plt.savefig('DimensionImpact_2rows.pdf', bbox_inches='tight')
plt.show()