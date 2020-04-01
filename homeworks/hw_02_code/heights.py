# ---------------------------------------------------------------------------- #
#                                                                              #
# SML@UniTs Course, Spring 2020 ~ Homework Solutions (code), nr. 2             #
#                                                                              #
# |> The "Male Heights" example <|                                             #
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
import seaborn as sns

import torch as th
import pyro
from pyro import distributions

pyro.set_rng_seed(1)

# ---------- #
# GIVEN DATA #
# ---------- #
my_mean = 180.0

prioalpha = 38.0
priobeta = 1110.0

my_invgamma = distributions.InverseGamma(prioalpha, priobeta)

my_sample = np.array(
    [
        183.0,
        173.0,
        181.0,
        170.0,
        176.0,
        180.0,
        187.0,
        176.0,
        171.0,
        190.0,
        184.0,
        173.0,
        176.0,
        179.0,
        181.0,
        186.0,
    ]
)

# --------- #
# FUNCTIONS #
# --------- #


# ---- #
# ELAB #
# ---- #

samples = my_invgamma([5000]).numpy()
foo = np.count_nonzero(np.logical_and(samples >= 22.0, samples < 41.0)) / 5000

newalpha = prioalpha + (np.size(my_sample) / 2.0)


msdev = np.sum((my_sample - my_mean) ** 2) / np.size(my_sample)
newbeta = priobeta + (np.size(my_sample) / 2.0) * msdev

posterior = my_invgamma = distributions.InverseGamma(newalpha, newbeta)

print(prioalpha, priobeta)
print(newalpha, newbeta)

my_invgamma = distributions.InverseGamma(prioalpha, priobeta)
posterior = distributions.InverseGamma(newalpha, newbeta)

samples_prio = my_invgamma([10000]).numpy()
sample_poste = posterior([10000]).numpy()

sns.distplot(my_invgamma([5000]).numpy(), hist=False, rug=False)
sns.distplot(posterior([5000]).numpy(), hist=False, rug=False)

plt.show()


# --------- #
# FUNCTIONS #
# --------- #


# ------------- #
# TEST-DRIVER 1 #
# ------------- #
print(foo)
