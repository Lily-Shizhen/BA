import numpy as np
import matplotlib.pyplot as plt
import pickle
import pathlib

##############################################################################
# 1) Choose your parameters & load results
##############################################################################

# Dimension list with 5 entries => 5 columns
d_list = [16, 32, 64, 128, 256]
demon_list = [0,1,2,4,8,16,32,64,128,256,512,1024,2048,4096,8192,16384]
selected_ks = [1, 8, 32, 128, 16384]

save_folder = 'results/'
path = pathlib.Path(save_folder)
results_file = 'results.pickle'

##############################################################################
# 2) Load the data from results.pickle
##############################################################################
with open(results_file, 'rb') as handle:
    results = pickle.load(handle)

##############################################################################
# 3) Gather risk/upper-bound data for each dimension across selected k-values
##############################################################################
dimension_loss_by_d = {}
dimension_bound_by_d = {}

valid_k = []
for k_val in selected_ks:
    if k_val in demon_list:
        valid_k.append(k_val)
    else:
        print(f"Warning: k={k_val} not found in demon_list; skipping.")

x_k = np.array(valid_k)  # for plotting on the x-axis

for d in d_list:
    losses = []
    bounds = []
    for k_val in valid_k:
        idx = demon_list.index(k_val)
        losses.append(results[d]['l_loss'][idx])
        bounds.append(results[d]['l_upper'][idx])
    dimension_loss_by_d[d] = np.array(losses)
    dimension_bound_by_d[d] = np.array(bounds)

##############################################################################
# 4) Determine the row-wide y-limits for the top row (ICL Risk),
#    and cap the top row's maximum y-limit at 9.
##############################################################################
risk_min = float('inf')
for d in d_list:
    val_min = dimension_loss_by_d[d].min()
    if val_min < risk_min:
        risk_min = val_min

risk_ymin = risk_min
risk_ymax = 9  # cap at 9 as requested

##############################################################################
# 5) We will NOT unify the second row's y-limits. Instead, we'll use a log-scale
#    so each dimension auto-scales for the risk upper bound subplot.
##############################################################################

##############################################################################
# 6) Create subplots: 2 rows × 5 columns
##############################################################################
fig, axs = plt.subplots(nrows=2, ncols=len(d_list), figsize=(4.0 * len(d_list), 8.0))

# Define x-ticks for powers of two: [2^0, 2^4, 2^8, 2^12, 2^16]
xticks = [1, 16, 256, 4096, 65536]
xtick_labels = [r'$2^0$', r'$2^4$', r'$2^8$', r'$2^{12}$', r'$2^{16}$']

for i, d in enumerate(d_list):
    ########################################################################
    # Top row: ICL Risk for dimension d (linear y-axis, shared scale [risk_ymin,9])
    ########################################################################
    ax_risk = axs[0, i]
    ax_risk.plot(
        x_k,
        dimension_loss_by_d[d],
        marker='o',
        linewidth=3,
        color='darkblue',
        label=f'd={d}'
    )
    # Shared y-limits for the top row
    ax_risk.set_ylim(risk_ymin, risk_ymax)

    # Log scale on x-axis => equal spacing for powers of two
    ax_risk.set_xscale('log', base=2)
    ax_risk.set_xticks(xticks)
    ax_risk.set_xticklabels(xtick_labels)

    ax_risk.set_title(f'Risk (d={d})', fontsize=12)
    if i == 0:
        ax_risk.set_ylabel('ICL Risk', fontsize=12)
    ax_risk.set_xlabel('k', fontsize=10)
    ax_risk.tick_params(axis='both', labelsize=10)

    if i == 0:
        ax_risk.legend(fontsize=10)

    ########################################################################
    # Bottom row: ICL Bound for dimension d (log-scale y-axis, auto-scale)
    ########################################################################
    ax_bound = axs[1, i]
    ax_bound.plot(
        x_k,
        dimension_bound_by_d[d],
        marker='s',
        linestyle='--',
        linewidth=3,
        color='cyan',
        label=f'd={d}'
    )
    # Use log scale on the y-axis so small changes are more apparent
    ax_bound.set_yscale('log', base=10)  # Using base-10 for the y-axis

    # Also keep log-scale on x-axis for consistent spacing
    ax_bound.set_xscale('log', base=2)
    ax_bound.set_xticks(xticks)
    ax_bound.set_xticklabels(xtick_labels)

    ax_bound.set_title(f'Bound (d={d})', fontsize=12)
    if i == 0:
        ax_bound.set_ylabel('ICL Bound', fontsize=12)
    ax_bound.set_xlabel('k', fontsize=10)
    ax_bound.tick_params(axis='both', labelsize=10)

    if i == 0:
        ax_bound.legend(fontsize=10)

plt.tight_layout()
plt.savefig('Fig3.16_risk_vs_k_10.pdf', bbox_inches='tight')
plt.show()