import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from setting3 import Regular4, D, PriorProcesser

setproctitle.setproctitle('BayesianSimulation')

matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')

import time
# import torch
import random
import numpy as np


# from torch.distributions.multivariate_normal import MultivariateNormal

def grid_position(row, col, total_cols=4):
    return total_cols * row + col + 1


if 1:  # define parameters
    S = 64
    num_k = 128
    k_list = [1, 2, 4, 8, 16, 32, 64, 128]
    k_list = [a - 1 for a in k_list]
    k_array = np.array(k_list)
    k_list = [r'${}$'.format(a) for a in k_list]
    K = 5000
    z = 1
    linewidth = 3.0

    pi_min = 0
    pi_max = 1
    pi_y_ticks = [0, 0.5, 1.0]
    pi_y_ticklabels = [r'$0.0$', r'$0.5$', r'$1.0$']
    dis_min = 0
    dis_max = 2
    err_min = 0
    err_max = 1

    legend_dis = 1.8

    base = 9
    # sm_list = [(1/base)**1.5, 1/base, (1/base)**0.5, 1, (base)**0.5, base, (base)**1.5]
    sm_list = [1 / base, (1 / base) ** 0.5, 1, (base) ** 0.5, base]
    sm_strs = [
        f'$1/81$',
        f'$1/9$',
        f'$1$',
        f'$9$',
        f'$81$',
    ]
    sw_list = [1 / base, (1 / base) ** 0.5, 1, (base) ** 0.5, base]
    sw_strs = [
        f'$1/81$',
        f'$1/9$',
        f'$1$',
        f'$9$',
        f'$81$',
    ]
    fig_name = 'Prediction'
    colors = ['#CD1818', '#0E21A0', '#116D6E', '#331D2C']
    M = 4

    plt.rcParams.update({'font.size': 64})

if 0:  # preprocess (disabled by default)
    for index, (sm, sw) in enumerate(zip(sm_list, sw_list)):
        folder = 'data/' + fig_name + '/sm=' + str(sm) + ' sw=' + str(sw) + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)

        print('\n***** ' + str(index) + ' *****')
        prior = Regular4(sm=sm, sw=sw, d=100)
        print('\n prior.delta_m = ', prior.delta_m)
        print('\n prior.delta_w = ', prior.delta_w)
        priorProcesser = PriorProcesser(prior)

        if 1:
            print('***************')
            colors = ['#CD1818', '#0E21A0', '#116D6E', '#331D2C']
            if 0:
                bs_xs, bs_ys, bs_retrieval, bs_learning = priorProcesser.draw_demon_sequences(K, num_k)
                B_preds = np.zeros([K, num_k])
                B_tpis = np.zeros([K, num_k, prior.M])
                B_ttopic_ms = np.zeros([K, num_k, prior.M, prior.d])
                B_ttopic_ws = np.zeros([K, num_k, prior.M, prior.d])
                B_adv_m = np.zeros([K, num_k, prior.M])
                B_adv_w = np.zeros([K, num_k, prior.M])
                B_tw = np.zeros([K, num_k, prior.d])
                for k in tqdm(range(1, num_k + 1, 1)):
                    for j in range(K):
                        xs, ys = bs_xs[j][:k], bs_ys[j][:k]
                        returned_dict = priorProcesser.predict(xs, ys, priorProcesser.topic_ws[0])
                        B_preds[j, k - 1] = returned_dict['prediction']
                        B_tpis[j, k - 1] = returned_dict['tpis']
                        B_ttopic_ms[j, k - 1] = returned_dict['ttopic_ms']
                        B_ttopic_ws[j, k - 1] = returned_dict['ttopic_ws']
                        B_adv_m[j, k - 1] = returned_dict['adv_m']
                        B_adv_w[j, k - 1] = returned_dict['adv_w']
                        B_tw[j, k - 1] = returned_dict['tw']
                np.save(folder + 'bs_xs.npy', bs_xs)
                np.save(folder + 'bs_ys.npy', bs_ys)
                np.save(folder + 'bs_retrieval.npy', bs_retrieval)
                np.save(folder + 'bs_learning.npy', bs_learning)
                np.save(folder + 'B_preds.npy', B_preds)
                np.save(folder + 'B_tpis.npy', B_tpis)
                np.save(folder + 'B_ttopic_ms.npy', B_ttopic_ms)
                np.save(folder + 'B_ttopic_ws.npy', B_ttopic_ws)
                np.save(folder + 'B_adv_m.npy', B_adv_m)
                np.save(folder + 'B_adv_w.npy', B_adv_w)
                np.save(folder + 'B_tw.npy', B_tw)
            else:
                bs_xs = np.load(folder + 'bs_xs.npy')
                bs_ys = np.load(folder + 'bs_ys.npy')
                bs_retrieval = np.load(folder + 'bs_retrieval.npy')
                bs_learning = np.load(folder + 'bs_learning.npy')
                B_preds = np.load(folder + 'B_preds.npy')
                B_tpis = np.load(folder + 'B_tpis.npy')
                B_ttopic_ms = np.load(folder + 'B_ttopic_ms.npy')
                B_ttopic_ws = np.load(folder + 'B_ttopic_ws.npy')
                B_adv_m = np.load(folder + 'B_adv_m.npy')
                B_adv_w = np.load(folder + 'B_adv_w.npy')
                B_tw = np.load(folder + 'B_tw.npy')

# ---- Create the 4x5 figure grid
if 1:  # set 4x5
    resize = 6
    fig, axs = plt.subplots(4, 5, figsize=(10 * resize, 4 * resize), sharex=True, sharey='row')

# ---- Row 1: mixture weights
if 1:  # originally "if 1: # draw 1x5 for pi"
    row_index = 1
    for row, (sm, sw) in enumerate(zip(sm_list, sw_list)):
        folder = f'data/{fig_name}/sm={sm} sw={sw}/'
        print('\n***** ', row, '-', 0, ' *****')
        bs_xs = np.load(folder + 'bs_xs.npy')
        bs_ys = np.load(folder + 'bs_ys.npy')
        bs_retrieval = np.load(folder + 'bs_retrieval.npy')
        bs_learning = np.load(folder + 'bs_learning.npy')
        B_preds = np.load(folder + 'B_preds.npy')
        B_tpis = np.load(folder + 'B_tpis.npy')
        B_ttopic_ms = np.load(folder + 'B_ttopic_ms.npy')
        print(np.mean(B_ttopic_ms))
        B_ttopic_ws = np.load(folder + 'B_ttopic_ws.npy')
        B_adv_m = np.load(folder + 'B_adv_m.npy')
        B_adv_w = np.load(folder + 'B_adv_w.npy')

        prior = Regular4(sm=sm, sw=sw, d=100)

        ax = axs[row_index, row]
        for m in range(M):
            TV = B_tpis[:, k_array, m]
            means = np.mean(TV, axis=0)
            std_devs = np.std(TV, axis=0)
            margin_error = z * std_devs
            lower_bound = means - margin_error
            upper_bound = means + margin_error

            ax.plot(means, label=fr'$\tilde{{\pi}}_{{{m + 1}}}$', linewidth=linewidth, color=colors[m])
            ax.fill_between(range(len(means)), lower_bound, upper_bound, color=colors[m], alpha=0.15)

            ax.set_xticks(np.arange(len(k_list)))
            ax.set_xticklabels(k_list)
            ax.set_yticks(pi_y_ticks)
            ax.set_yticklabels(pi_y_ticklabels)
            ax.set_ylim([-err_min, err_max])

        if row == len(sm_strs) - 1:
            ax.legend(loc='right', bbox_to_anchor=(legend_dis, 0.5))

# ---- Row 0: distance of task retrieval
if 1:  # changed from "if 2:" to "if 1:"
    row_index = 0
    for row, (sm, sw) in enumerate(zip(sm_list, sw_list)):
        folder = f'data/{fig_name}/sm={sm} sw={sw}/'
        print('\n***** ', row, '-', 0, ' *****')
        bs_xs = np.load(folder + 'bs_xs.npy')
        bs_ys = np.load(folder + 'bs_ys.npy')
        bs_retrieval = np.load(folder + 'bs_retrieval.npy')
        bs_learning = np.load(folder + 'bs_learning.npy')
        B_preds = np.load(folder + 'B_preds.npy')
        B_tpis = np.load(folder + 'B_tpis.npy')
        B_ttopic_ms = np.load(folder + 'B_ttopic_ms.npy')
        B_ttopic_ws = np.load(folder + 'B_ttopic_ws.npy')
        B_adv_m = np.load(folder + 'B_adv_m.npy')
        B_adv_w = np.load(folder + 'B_adv_w.npy')

        prior = Regular4(sm=sm, sw=sw, d=100)

        ax = axs[row_index, row]
        for m in range(M):
            TV = B_ttopic_ws[:, k_array, m, :] - prior.new_task_w
            TV = np.sum(TV ** 2, axis=2) ** 0.5
            means = np.mean(TV, axis=0)
            std_devs = np.std(TV, axis=0)
            margin_error = z * std_devs
            lower_bound = means - margin_error
            upper_bound = means + margin_error

            ax.plot(means, label=fr'$\|\tilde{{\boldsymbol{{w}}}}_{{{m + 1}}} - \boldsymbol{{w}}^*\|$',
                    linewidth=linewidth, color=colors[m])
            ax.fill_between(range(len(means)), lower_bound, upper_bound, color=colors[m], alpha=0.15)

            ax.set_xticks(np.arange(len(k_list)))
            ax.set_xticklabels(k_list)
            ax.set_ylim([-dis_min, dis_max])
            ax.set_title(r'$\delta_{\mu}=\delta_{w}=$' + ' ' + sw_strs[row])

        if row == len(sm_strs) - 1:
            ax.legend(loc='right', bbox_to_anchor=(legend_dis, 0.5), fontsize=S * 0.8)

# ---- Row 2: distance of task learning
if 1:  # changed from "if 3:" to "if 1:"
    row_index = 2
    for row, (sm, sw) in enumerate(zip(sm_list, sw_list)):
        folder = f'data/{fig_name}/sm={sm} sw={sw}/'
        print('\n***** ', row, '-', 0, ' *****')
        bs_xs = np.load(folder + 'bs_xs.npy')
        bs_ys = np.load(folder + 'bs_ys.npy')
        bs_retrieval = np.load(folder + 'bs_retrieval.npy')
        bs_learning = np.load(folder + 'bs_learning.npy')
        B_preds = np.load(folder + 'B_preds.npy')
        B_tpis = np.load(folder + 'B_tpis.npy')
        B_ttopic_ms = np.load(folder + 'B_ttopic_ms.npy')
        B_ttopic_ws = np.load(folder + 'B_ttopic_ws.npy')
        B_adv_m = np.load(folder + 'B_adv_m.npy')
        B_adv_w = np.load(folder + 'B_adv_w.npy')
        B_tw = np.load(folder + 'B_tw.npy')

        prior = Regular4(sm=sm, sw=sw, d=100)

        ax = axs[row_index, row]

        TV = B_tw[:, k_array, :] - prior.new_task_w
        TV = np.sum(TV ** 2, axis=2) ** 0.5
        means = np.mean(TV, axis=0)
        std_devs = np.std(TV, axis=0)
        margin_error = z * std_devs
        lower_bound = means - margin_error
        upper_bound = means + margin_error

        ax.plot(np.arange(len(k_list)), means, 'o-',
                label=r'$\|\tilde{{\boldsymbol{{w}}}} - \boldsymbol{{w}}^*\|$',
                linewidth=linewidth, color='black')
        ax.fill_between(range(len(means)), lower_bound, upper_bound, color='black', alpha=0.15)

        ax.set_xticks(np.arange(len(k_list)))
        ax.set_xticklabels(k_list)
        ax.set_ylim([-dis_min, dis_max / 2])

        if row == len(sm_strs) - 1:
            ax.legend(loc='right', bbox_to_anchor=(legend_dis, 0.5), fontsize=S * 0.8)

# ---- Row 3: prediction error
if 1:  # changed from "if 4:" to "if 1:"
    row_index = 3
    for row, (sm, sw) in enumerate(zip(sm_list, sw_list)):
        folder = f'data/{fig_name}/sm={sm} sw={sw}/'
        print('\n***** ', row, '-', 0, ' *****')
        bs_xs = np.load(folder + 'bs_xs.npy')
        bs_ys = np.load(folder + 'bs_ys.npy')
        bs_retrieval = np.load(folder + 'bs_retrieval.npy')
        bs_learning = np.load(folder + 'bs_learning.npy')
        B_preds = np.load(folder + 'B_preds.npy')
        B_tpis = np.load(folder + 'B_tpis.npy')
        B_ttopic_ms = np.load(folder + 'B_ttopic_ms.npy')
        B_ttopic_ws = np.load(folder + 'B_ttopic_ws.npy')
        B_adv_m = np.load(folder + 'B_adv_m.npy')
        B_adv_w = np.load(folder + 'B_adv_w.npy')

        prior = Regular4(sm=sm, sw=sw, d=100)

        ax = axs[row_index, row]

        TV = (B_preds[:, k_array] - bs_learning[:, k_array]) ** 2
        means = np.mean(TV, axis=0)
        std_devs = np.std(TV, axis=0)
        margin_error = z * std_devs
        lower_bound = means - margin_error
        upper_bound = means + margin_error

        ax.plot(np.arange(len(k_list)), means, 'o-',
                label=r'$(\mathcal{F}^* - y_{k+1}^{*})^2$',
                linewidth=linewidth, color='black')
        ax.fill_between(range(len(means)), lower_bound, upper_bound, color='black', alpha=0.15)

        ax.set_xticks(np.arange(len(k_list)))
        ax.set_xticklabels(k_list)
        ax.set_ylim([-err_min, err_max])

        if row == len(sm_strs) - 1:
            ax.legend(loc='right', bbox_to_anchor=(legend_dis, 0.5), fontsize=S * 0.8)

if 1:
    fig.text(0.5, -0.00, 'Number of In-Context Examples ' + r'$(k)$',
             ha='center', va='center', fontsize=64)
    plt.tight_layout()
    plt.savefig('Figure3.2.pdf', bbox_inches='tight')