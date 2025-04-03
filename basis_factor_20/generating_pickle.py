import os
import numpy as np
import pickle
import pathlib

# Define the list of dimensions and the number of k subfolders (cutoff)
d_list = [16, 32, 64, 128, 256]
# (demon_list and k_list were used during preprocessing; here we assume the subfolders are named as k values.)
cut = 17  # number of k values (subfolders) to load per dimension

results = {}

save_folder = 'results'  # your root results folder
save_folder = os.path.join(os.getcwd(), save_folder)  # full path if needed

# Loop over each dimension folder
for d in d_list:
    results[d] = {}
    folder_d = os.path.join(save_folder, str(d))
    # Get the subfolder names sorted numerically
    k_folders = sorted(
        [sub for sub in os.listdir(folder_d) if os.path.isdir(os.path.join(folder_d, sub))],
        key=lambda x: int(x)
    )
    # Optionally, limit to the first "cut" subfolders
    k_folders = k_folders[:cut]

    l_loss_list = []
    l_upper_list = []
    tpis_list = []
    tw_list = []

    # For each k folder, load the corresponding .npy files
    for k_folder in k_folders:
        folder_path = os.path.join(folder_d, k_folder)
        l_loss = np.load(os.path.join(folder_path, 'learning_loss_update.npy'))
        l_upper = np.load(os.path.join(folder_path, 'learning_upper_update.npy'))
        tpis = np.load(os.path.join(folder_path, 'tpis_update.npy'))
        tw = np.load(os.path.join(folder_path, 'tw_update.npy'))

        l_loss_list.append(l_loss)
        l_upper_list.append(l_upper)
        tpis_list.append(tpis)
        tw_list.append(tw)

    # Save as arrays for this dimension
    results[d]['l_loss'] = np.array(l_loss_list)
    results[d]['l_upper'] = np.array(l_upper_list)
    results[d]['tpis'] = np.concatenate(tpis_list, axis=0)
    results[d]['tw'] = np.array(tw_list)

# Save the consolidated results dictionary to a pickle file
with open('results.pickle', 'wb') as handle:
    pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("results.pickle generated successfully!")