import sys, joblib
import numpy as np
import lasagne.layers as L


"""
a script to load network parameters from a .npz file to a .pkl file
does not enforce architectural compatibility between the two files,
the responsibility for that lies with the user
"""
if __name__=="__main__":
	args = sys.argv
	if len(args) != 3:
		sys.exit("expected usage: python unpackage_params.py <src> <dest>")
	src = args[1]
	dest = args[2]
	params = np.load(src)
	mean_params = params['mean_params']
	std_params = params['std_params']
	data = joblib.load(dest)
	L.set_all_param_values(data['policy']._l_mean, mean_params)
	L.set_all_param_values(data['policy']._l_log_std, std_params)
	joblib.dump(data, dest)