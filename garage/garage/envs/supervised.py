import time
import numpy as np
import tensorflow as tf
import joblib
import garage.tf.core.layers as L

import sys
import os


alpha = 1e-4
beta = 1e2
minibatch_size = 2000
num_epochs = 10000
eps = 7e-3

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	indices = np.arange(len(targets))
	if shuffle:
		np.random.shuffle(indices)
	for start_idx in range(0, indices.size, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


def initialize_tf_vars(sess):
        """Initialize all uninitialized variables in session."""
        uninit_set = set(
            e.decode()
            for e in sess.run(tf.report_uninitialized_variables()))
        sess.run(
            tf.variables_initializer([
                v for v in tf.global_variables()
                if v.name.split(':')[0] in uninit_set
            ]))


def load_data(filepath):
	data = np.load(filepath)
	obs = data['obs']
	actions = data['act']

	assert obs.shape[0] ==  actions.shape[0]
	return obs, actions

def softplus(x):
	return np.log(np.exp(x) + 1)

if __name__ == '__main__':
	print("prepare data")
	folder = str()

	print("prepare network")

	sess = tf.Session()
	src = "/home/tbla2725/garage/data/local/experiment/networks/untrained_rrt_mountaincar1.pkl"
	dest = "/home/tbla2725/garage/data/local/experiment/networks/trained_sst_mountaincar1.pkl"
	with sess.as_default(): data = joblib.load(src)
	policy = data['policy']
	print("prepare data")
	obs, act = load_data('../data/mountaincar/sst_trajectories.npz')
	X = obs
	y = act
	y = np.clip(y, -1.0, 1.0)
	input_var = tf.placeholder(tf.float32, shape=[None, X.shape[1]], name='inputs')
	target_var = tf.placeholder(tf.float32, shape=[None, y.shape[1]], name='targets')
	#X = (X - X.mean())/(X.std() + 1e-8)
	#y = (y - y.mean())/(y.std() + 1e-8)
	print("prepare training")
	dist = policy.distribution
	dist_info = policy.dist_info_sym(input_var)
	likelihood = dist.log_likelihood_sym(target_var, dist_info)
	policy_outputs = list(policy._outputs.values())
	print(policy_outputs)
	policy_input = policy._obs_var
	pred = policy.predict(input_var)
	pred_error = tf.reduce_mean(tf.square(pred - target_var), axis=0)
	l2_regularization = L.regularize_network_params(policy_outputs[0], tf.nn.l2_loss,
		{'regularizable': True})
	print("prepare objective function")
	loss = tf.reduce_mean(pred_error) \
			+ 1e-4 * l2_regularization#- tf.reduce_mean(likelihood) \
	print("prepare training function")
	inputs_adv_var = tf.stop_gradient(
		tf.clip_by_value(
			input_var + eps * tf.sign(tf.gradients(loss, input_var)[0]),
			-1.0,
			1.0
		)
	)
	train_op = tf.train.AdamOptimizer().minimize(loss, name="train_op")
	print("initialize variables")
	initialize_tf_vars(sess)
	print("begin training...")
	for epoch in range(num_epochs):
		if epoch % 50 == 0 and epoch != 0:
			with sess.as_default(): joblib.dump(data, dest)
		rate = 1.0
		train_err = 0
		eval_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X, y, minibatch_size, shuffle=True):
			inputs, targets = batch
			#inputs_adv = sess.run([inputs_adv_var],
				#feed_dict={input_var : inputs, target_var : targets})[0]
			#inputs = np.concatenate([inputs, inputs_adv], axis=0)
			#targets = np.concatenate([targets, targets], axis=0)
			_, batch_train_err, batch_mse, batch_pred = sess.run(
				[train_op, loss, pred_error, pred],
				feed_dict={input_var : inputs, target_var : targets}
			)
			train_err += batch_train_err
			eval_err += batch_mse
			train_batches += 1
	
		reg_err = sess.run(l2_regularization)
		print("Epoch {} of {} took {:.3f}s".format(
			epoch+1, num_epochs, time.time() - start_time))
		print("\tl2 regularization loss:\t{}".format(reg_err))
		print("\ttraining loss:\t\t{:.6f}".format(train_err / train_batches))
		print("\tpred error:\t\t{}".format(eval_err / train_batches))
	with sess.as_default(): joblib.dump(data, dest)
	sys.exit()
