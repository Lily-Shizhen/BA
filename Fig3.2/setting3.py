import os
import argparse
import setproctitle
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
import numpy as np
import scipy
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from numpy.linalg import eig
from decimal import Decimal, getcontext

import numpy as np


def sorted_eigenvalues(matrix):
    eigenvalues = np.linalg.eigh(matrix)[0]
    return sorted(eigenvalues, reverse=True)


def numpy_to_decimal(arr, precision=2000):
    getcontext().prec = precision
    vec_decimal = np.vectorize(Decimal)
    return vec_decimal(arr)


def decimal_to_numpy(decimal_arr, dtype=float):
    vec_float = np.vectorize(lambda x: dtype(str(x)))
    return vec_float(decimal_arr)


getcontext().prec = 50

# python train.py --name=GPT
matplotlib.rc('text', usetex=True)
matplotlib.rc('text.latex', preamble=r'\usepackage{amsmath}')


class SampleGenerator:
    def __init__(self, d, m_center, w_center, s):
        self.d = d
        self.m_center = m_center.copy()
        self.w_center = w_center.copy()
        self.s = s

    def generate_xy_samples(self, k):
        xs = np.random.multivariate_normal(mean=self.m_center, cov=self.s * np.eye(self.d), size=k)
        ys = xs @ self.w_center
        return xs, ys[:, np.newaxis]


class D:  # define GMM prior with D_num components
    def __init__(self, D_num=4, sm=0.25, sw=0.25, rotate=0, due=False, case=None):
        self.d = D_num  # dimension = components = 4
        self.M = D_num * (1 + due)

        self.topic_ms = np.zeros([self.M, D_num])  # the mean vector of each topic/component
        for d in range(D_num):
            self.topic_ms[d, d] = 1
            if due == True:
                self.topic_ms[D_num + d, d] = -1
        self.topic_ws = self.topic_ms.copy()  # Associated efficient vector for the mean vector. To keep the output weight distribution and the input distribution start with same pattern

        self.pis = np.ones([self.M]) / self.M  # weights for each component/proportion of input data
        self.s = 1  # Scale for input x/Normalization factors for input noise
        self.x_covariance = self.s ** 2 * np.eye(self.d)  # the variance of each dimension of  x  is  unit variance
        self.t = 1  # Scale for Target Variance./Normalization factors for output noise
        self.y_variance = self.t ** 2  # the noise in  y  is Gaussian with unit variance
        self.sm = sm  # standard deviation of input means(
        self.m_covariance = self.sm ** 2 * np.eye(self.d)
        self.sw = sw  # Standard deviation of the prior weights/mapping parameters (topic_ws)
        self.w_covariance = self.sw ** 2 * np.eye(self.d)
        # delta_m is a weighting factor that determines how much the prior means (topic_ms) contribute to the overall model compared to the observed data.
        self.delta_m = (self.sm / self.s) ** 2  # Relative input variance
        # delta_w is the relative strength of the prior weights contribute to the overall model’s predictions.
        self.delta_w = (self.sw / self.t) ** 2  # Relative weight variance.

        # generate new tasks by sampling miu and w
        # by default, linear combination of normalized miu and w
        if case == None:
            new_task_m = 3 * self.topic_ms[0] + self.topic_ms[1]
            self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
            new_task_w = 3 * self.topic_ws[0] + self.topic_ws[1]
            self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
        elif case == 1:
            new_task_m = 1 * self.topic_ms[0] + 1 * self.topic_ms[1]
            self.new_task_m = new_task_m / np.linalg.norm(new_task_m)
            new_task_w = 1 * self.topic_ws[0] + 1 * self.topic_ws[1]
            self.new_task_w = new_task_w / np.linalg.norm(new_task_w)
        # random perturbations around miu and w
        elif case == 'random':
            index = np.random.choice(np.arange(self.M))
            chosen_center_m = self.topic_ms[index]
            chosen_center_w = self.topic_ws[index]
            self.new_task_m = np.random.multivariate_normal(mean=self.topic_ms[index],
                                                            cov=self.sm ** 2 * np.eye(self.d))
            self.new_task_w = np.random.multivariate_normal(mean=self.topic_ws[index],
                                                            cov=self.sw ** 2 * np.eye(self.d))


class Regular4:
    """
    Minimal-change version of Regular4 that can handle any dimension 'd'.
    If d=3, it uses the old 3D tetrahedron logic. If d!=3, it creates random 4 centers in R^d.
    We also remove the 3D 'visualize' method to avoid tetrahedron plotting code.
    """

    def __init__(self, sm=0.25, sw=0.25, match=True, ratio=None, colors=None, d=3):
        self.d = d
        self.M = 4
        self.sm = sm
        self.sw = sw
        self.colors = colors

        if self.d == 3:
            # Original 3D tetrahedron code:
            self.topic_ms = np.array([
                [0, 0, -1],
                [(8 / 9) ** 0.5, 0, 1 / 3],
                [-(2 / 9) ** 0.5, (2 / 3) ** 0.5, 1 / 3],
                [-(2 / 9) ** 0.5, -(2 / 3) ** 0.5, 1 / 3]
            ])
            self.topic_ms = self.topic_ms[:, [0, 2, 1]]
            self.topic_ws = self.topic_ms.copy()
            if not match:
                self.topic_ws = self.topic_ws[[1, 2, 3, 0]]
        else:
            # For d != 3, create random centers in R^d.
            rng = np.random.default_rng(seed=123)  # optional fixed seed for reproducibility
            self.topic_ms = rng.normal(size=(self.M, self.d))
            for i in range(self.M):
                norm_val = np.linalg.norm(self.topic_ms[i])
                if norm_val > 0:
                    self.topic_ms[i] /= norm_val

            self.topic_ws = rng.normal(size=(self.M, self.d))
            for i in range(self.M):
                norm_val = np.linalg.norm(self.topic_ws[i])
                if norm_val > 0:
                    self.topic_ws[i] /= norm_val

        # mixture weights
        self.pis = np.ones([self.M]) / self.M

        # other parameters
        self.s = 1
        self.x_covariance = self.s ** 2 * np.eye(self.d)
        self.t = 1
        self.y_variance = self.t ** 2
        self.m_covariance = (self.sm ** 2) * np.eye(self.d)
        self.w_covariance = (self.sw ** 2) * np.eye(self.d)
        self.delta_m = (self.sm / self.s) ** 2
        self.delta_w = (self.sw / self.t) ** 2

        # Define a "new task" vector
        if ratio is None:
            # Minimal approach: combine topic_ms[0] + small fraction of topic_ms[1]
            new_task_m = self.topic_ms[0] + 0.2 * self.topic_ms[1]
            norm_val = np.linalg.norm(new_task_m)
            if norm_val > 0:
                new_task_m /= norm_val
            self.new_task_m = new_task_m

            new_task_w = self.topic_ws[0] + 0.2 * self.topic_ws[1]
            norm_val = np.linalg.norm(new_task_w)
            if norm_val > 0:
                new_task_w /= norm_val
            self.new_task_w = new_task_w
        else:
            # Keep your original ratio logic if you wish
            pass


class PriorProcesser:
    def __init__(self, prior):
        self.d = prior.d
        self.M = prior.M
        self.topic_ms = prior.topic_ms
        self.topic_ws = prior.topic_ws
        self.pis = prior.pis
        self.s = prior.s
        self.x_covariance = prior.x_covariance
        self.t = prior.t
        self.y_variance = prior.y_variance
        self.sm = prior.sm
        self.m_covariance = prior.m_covariance
        self.sw = prior.sw
        self.w_covariance = prior.w_covariance
        self.delta_m = prior.delta_m
        self.delta_w = prior.delta_w

        self.new_task_m = prior.new_task_m
        self.new_task_w = prior.new_task_w

    def draw_topic(self):
        index = np.random.choice(self.M, p=self.pis)
        return self.topic_ms[index], self.topic_ws[index]

    def draw_topics(self, num):
        indexes = np.random.choice(self.M, p=self.pis, size=num)
        return self.topic_ms[indexes], self.topic_ws[indexes]

    def draw_task(self, topic_m, topic_w):
        task_m = np.random.multivariate_normal(topic_m, self.m_covariance)
        task_w = np.random.multivariate_normal(topic_w, self.w_covariance)
        return task_m, task_w

    def draw_tasks(self, topic_m, topic_w, num):
        task_ms = np.random.multivariate_normal(topic_m, self.m_covariance, size=num)
        task_ws = np.random.multivariate_normal(topic_w, self.w_covariance, size=num)
        return task_ms, task_ws

    def draw_sequence(self, k):
        topic_m, topic_w = self.draw_topic()
        task_m, task_w = self.draw_task(topic_m, topic_w)
        xs = np.random.multivariate_normal(task_m, self.x_covariance, size=k + 1)
        ys = xs @ task_w + np.random.normal(0, self.y_variance, size=k + 1)
        return xs, ys

    def draw_sequences(self, bs, k):
        topic_ms, topic_ws = self.draw_topics(bs)
        bs_xs = []
        bs_ys = []
        bs_zs = []
        for i in range(bs):
            task_m, task_w = self.draw_task(topic_ms[i], topic_ws[i])
            xs = np.random.multivariate_normal(task_m, self.x_covariance, size=k)
            bs_xs.append(xs)
            zs = xs @ task_w
            bs_zs.append(zs)
            ys = zs + np.random.normal(0, self.y_variance, size=k)
            bs_ys.append(ys)
        bs_xs = np.stack(bs_xs, axis=0)
        bs_ys = np.stack(bs_ys, axis=0)
        bs_zs = np.stack(bs_zs, axis=0)
        return bs_xs, bs_ys, bs_ys

    def draw_demon_sequences(self, bs, k):
        # "demonstration" sequences for the same new_task_m, new_task_w
        topic_ms, topic_ws = self.draw_topics(bs)
        bs_xs = []
        bs_ys = []
        bs_retrieval = []
        bs_learning = []
        for i in range(bs):
            xs = np.random.multivariate_normal(self.new_task_m, self.x_covariance, size=k)
            bs_xs.append(xs)
            retrieval = xs @ self.topic_ws[0]
            learning = xs @ self.new_task_w
            bs_retrieval.append(retrieval)
            bs_learning.append(learning)
            ys = learning[:, np.newaxis]
            bs_ys.append(ys)
        bs_xs = np.stack(bs_xs, axis=0)
        bs_ys = np.stack(bs_ys, axis=0)
        bs_retrieval = np.stack(bs_retrieval, axis=0)
        bs_learning = np.stack(bs_learning, axis=0)
        return bs_xs, bs_ys, bs_retrieval, bs_learning

    def predict(self, xs, ys, target_w=None):
        k = xs.shape[0] - 1
        I = np.eye(self.d)
        tpis = self.pis.copy()
        ttopic_ms = self.topic_ms.copy()
        ttopic_ws = self.topic_ws.copy()

        D_m = self.delta_m * np.sum(xs, axis=0)
        Dm_I = self.delta_m * (k + 1) * I
        IDmI_inv = np.linalg.inv(I + Dm_I)

        D_w = self.delta_w * np.sum(xs[:-1] * ys[:-1], axis=0)
        Dw_I = self.delta_w * np.sum(xs[:-1, :, np.newaxis] * xs[:-1, np.newaxis, :], axis=0)
        IDwI_inv = np.linalg.inv(I + Dw_I)

        adv_m = np.zeros([self.M])
        adv_w = np.zeros([self.M])
        for b in range(self.M):
            numer_m = (
                    self.topic_ms[b] @ self.topic_ms[b]
                    - (self.topic_ms[b] + D_m) @ IDmI_inv @ (self.topic_ms[b] + D_m).T
            )
            denom_m = 2 * (self.sm ** 2)
            adv_m[b] = numer_m / denom_m

            numer_w = (
                    self.topic_ws[b] @ self.topic_ws[b]
                    - (self.topic_ws[b] + D_w) @ IDwI_inv @ (self.topic_ws[b] + D_w).T
            )
            denom_w = 2 * (self.sw ** 2)
            adv_w[b] = numer_w / denom_w

            # updated topic centers
            ttopic_ms[b] = IDmI_inv @ (self.topic_ms[b] + D_m)
            ttopic_ws[b] = IDwI_inv @ (self.topic_ws[b] + D_w)

        # shift relative to the first component
        adv_m = adv_m - adv_m[0]
        adv_w = adv_w - adv_w[0]
        adv = adv_m + adv_w

        import scipy
        tpis = tpis * scipy.special.softmax(-adv)
        tpis = tpis / np.sum(tpis)

        prediction = tpis @ (ttopic_ws @ xs[-1])
        tw = tpis @ ttopic_ws

        l_loss = (prediction - ys[-1]) ** 2
        if target_w is None:
            r_loss = None
        else:
            r_loss = (prediction - target_w @ xs[-1]) ** 2

        return {
            'l_loss': l_loss,
            'r_loss': r_loss,
            'tw': tw,
            'prediction': prediction,
            'tpis': tpis,
            'ttopic_ms': ttopic_ms,
            'ttopic_ws': ttopic_ws,
            'retrieval': target_w @ xs[-1] if target_w is not None else None,
            'learning': ys[-1],
            'adv_m': adv_m,
            'adv_w': adv_w,
        }