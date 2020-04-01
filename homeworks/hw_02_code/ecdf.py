# ---------------------------------------------------------------------------- #
#                                                                              #
# SML@UniTs Course, Spring 2020 ~ Homework Solutions (code), nr. 2             #
#                                                                              #
# |> Implementation of an eCDF function <|                                     #
#                                                                              #
# (C) 2020-* Emanuele Ballarin <emanuele@ballarin.cc>                          #
# (C) 2020-* AI-CPS@UniTS Laboratory (a.k.a. Bortolussi Group)                 #
#                                                                              #
# Distribution: MIT License                                                    #
# (Full text: https://github.com/emaballarin/RADLER/blob/master/LICENSE)       #
#                                                                              #
# Eventually-updated version:                                                  #
# https://github.com/emaballarin/statistical-machine-learning                  #
#                                                                              #
# ---------------------------------------------------------------------------- #


# ------- #
# IMPORTS #
# ------- #

import numpy as np
import matplotlib.pyplot as plt

import torch as th
import pyro
from pyro import distributions

pyro.set_rng_seed(1)


# --------- #
# FUNCTIONS #
# --------- #


def cdf(dist, x, samplesize=5000):
    return float(np.count_nonzero(dist([samplesize]).numpy() <= x)) / samplesize


def plot_histcdf(dist, samplesize=5000, unif_bins=250):
    normcounts, edges = np.histogram(dist([samplesize]), unif_bins, density=False)
    plt.plot(edges[1:], np.cumsum(normcounts) / np.sum(normcounts))
    plt.show()


# ------------- #
# TEST-DRIVER 1 #
# ------------- #

foo = distributions.Uniform(0.0, 10.0)
print(cdf(foo, 5.0))

# ------------- #
# TEST-DRIVER 2 #
# ------------- #
bar = distributions.Normal(0.0, 1.0)
plot_histcdf(bar)
