import numpy as np
import matplotlib.pyplot as plt
import polynomial_chaos.poly as poly
import polynomial_chaos.poly_chaos_distributions as ch
import polynomial_chaos.distributions as dst
from scipy.integrate import quad
from util.storage import load_and_show_fig

load_and_show_fig("../data/interpol_invmat_trial5_512_0.00005.pickle")
