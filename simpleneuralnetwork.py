import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

plt.style.use('./deeplearning.mplstyle')

from lab_utils_common import dlc, sigmoid
from lab_coffee_utils import load_coffee_data, plt_roast, plt_prob, plt_layer, plt_network, plt_output_unit

logging.getLogger("tensorflow").setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)

X, y = load_data()
# plt_roast(X,Y)

#print(f"Temperature Max, Min pre normalization: {np.max(X[:,0]):0.2f}, {np.min(X[:,0]):0.2f}")
#print(f"Duration    Max, Min pre normalization: {np.max(X[:,1]):0.2f}, {np.min(X[:,1]):0.2f}")
#norm_l = tf.keras.layers.Normalization(axis=-1)
#norm_l.adapt(X)  # learns mean, variance
#Xn = norm_l(X)
#print(f"Temperature Max, Min post normalization: {np.max(Xn[:,0]):0.2f}, {np.min(Xn[:,0]):0.2f}")
#print(f"Duration    Max, Min post normalization: {np.max(Xn[:,1]):0.2f}, {np.min(Xn[:,1]):0.2f}")
print(np.sum([[[0, 1], [2,2], [3,3]], [[6, 6], [7, 7], [8, 8]]], axis=0))
#print(layer(input_data))