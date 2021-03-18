import sys, joblib
import numpy as np
import garage.tf.core.layers as L
import tensorflow as tf


"""
a script to save network parameters from a .pkl file with GPUarrays
 to a .npz file that can be opened without needing a gpu
"""

if __name__=="__main__":
	args = sys.argv
	if len(args) != 3:
		sys.exit("expected usage: python package_params.py <src> <dest>")
	src = args[1]
	dest = args[2]
	sess = tf.Session()
	with sess.as_default():
		data = joblib.load(src)
		mean_params = L.get_all_param_values(data['policy']._l_mean)
		std_params = L.get_all_param_values(data['policy']._std_network._l_out)
	np.savez_compressed(dest, mean_params=mean_params, std_params=std_params)
