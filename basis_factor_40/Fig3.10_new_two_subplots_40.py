import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

##############################################################################
# 1) Choose your parameters & load results
##############################################################################

d_list = [16, 32, 64, 128, 256]  # Example dimension list
demon_list = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
selected_ks = [1,4,16,64,256,1024,4096,16384]

save_folder = 'results/'
path = pathlib.Path(save_folder)
results_file = 'results.pickle'

##############################################################################
# 2) Load the data from results.pickle
##############################################################################
with open(results_file, 'rb') as handle:
    results = pickle.load(handle)

##############################################################################
# 3) Gather risk/upper data in the form dimension_loss[k] and dimension_upper[k]
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
        all_loss = results[d]['l_loss']
        all_upper = results[d]['l_upper']
        losses_for_this_k.append(all_loss[idx])
        uppers_for_this_k.append(all_upper[idx])

    dimension_loss[k_val] = np.array(losses_for_this_k)
    dimension_upper[k_val] = np.array(uppers_for_this_k)

##############################################################################
# 4) Reshape data so we can plot:
#    - x-axis = list of k-values (log scale),
#    - one line per dimension
##############################################################################

valid_k = [k for k in selected_ks if k in dimension_loss]

dimension_loss_by_d = {d: [] for d in d_list}
dimension_upper_by_d = {d: [] for d in d_list}

for k_val in valid_k:
    risks_for_k = dimension_loss[k_val]
    uppers_for_k = dimension_upper[k_val]
    for i, d in enumerate(d_list):
        dimension_loss_by_d[d].append(risks_for_k[i])
        dimension_upper_by_d[d].append(uppers_for_k[i])

for d in d_list:
    dimension_loss_by_d[d] = np.array(dimension_loss_by_d[d])
    dimension_upper_by_d[d] = np.array(dimension_upper_by_d[d])

##############################################################################
# 5) Create two subplots side by side:
##############################################################################
fig, (ax_risk, ax_upper) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

x_k = np.array(valid_k)

# --- Left subplot: ICL Risk vs. k-values (LOG x-axis) ---
for d in d_list:
    ax_risk.plot(x_k, dimension_loss_by_d[d], marker='o', linewidth=2, label=f'd={d}')
ax_risk.set_title('ICL Risk', fontsize=14)
ax_risk.set_xlabel('k-value', fontsize=14)
ax_risk.set_ylabel('Risk', fontsize=14)
ax_risk.legend(fontsize=12)
ax_risk.tick_params(axis='both', labelsize=12)

# Set x-scale to log
ax_risk.set_xscale('log', base=2)  # or base=10 if you prefer
# Optionally, pick nice tick locations
# For example, if valid_k covers [1, 2, 4, 8, 16, 32, 64, 128], you can do:
ax_risk.set_xticks([1, 4, 16, 64, 256, 1024, 4096, 16384])
ax_risk.set_xticklabels([r'$2^0$', r'$2^2$', r'$2^4$', r'$2^6$', r'$2^8$', r'$2^{10}$', r'$2^{12}$', r'$2^{14}$'], rotation=30)

# --- Right subplot: ICL Upper Bound vs. k-values (LOG x-axis) ---
for d in d_list:
    ax_upper.plot(x_k, dimension_upper_by_d[d], marker='s', linestyle='--', linewidth=2, label=f'd={d}')
ax_upper.set_title('ICL Risk Upper Bound', fontsize=14)
ax_upper.set_xlabel('k-value', fontsize=14)
ax_upper.set_ylabel('Upper Bound', fontsize=14)
ax_upper.legend(fontsize=12)
ax_upper.tick_params(axis='both', labelsize=12)

# Set x-scale to log
ax_upper.set_xscale('log', base=2)
ax_upper.set_xticks([1, 4, 16, 64, 256, 1024, 4096, 16384])
ax_upper.set_xticklabels([r'$2^0$', r'$2^2$', r'$2^4$', r'$2^6$', r'$2^8$', r'$2^{10}$', r'$2^{12}$', r'$2^{14}$'], rotation=30)

plt.tight_layout()
plt.savefig('Fig3.10_subplots_logX_40.pdf', bbox_inches='tight')
plt.show()