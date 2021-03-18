import numpy as np
import tensorflow as tf
import inspect

from garage.core import Serializable
from garage.tf.core import layers as ly
from garage.tf.core import LayersPowered
from garage.tf.misc.tensor_utils import initialize_tf_vars

class MLP(LayersPowered, Serializable):
    def __init__(
            self,
            output_dim,
            hidden_sizes,
            hidden_nonlinearity,
            output_nonlinearity,
            name=None,
            hidden_w_init=ly.XavierUniformInitializer(),
            hidden_b_init=tf.zeros_initializer(),
            output_w_init=ly.XavierUniformInitializer(),
            output_b_init=tf.zeros_initializer(),
            input_var=None,
            input_layer=None,
            input_shape=None,
            batch_normalization=False,
            weight_normalization=False,
            trainable=True,
    ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name, "MLP"):
            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, ) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_hid = l_in
            if batch_normalization:
                l_hid = ly.batch_norm(l_hid)
            for idx, hidden_size in enumerate(hidden_sizes):
                l_hid = ly.DenseLayer(
                    l_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="hidden_%d" % idx,
                    w=hidden_w_init,
                    b=hidden_b_init,
                    weight_normalization=weight_normalization)
                if batch_normalization:
                    l_hid = ly.batch_norm(l_hid)
                self._layers.append(l_hid)
            l_out = ly.DenseLayer(
                l_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                w=output_w_init,
                b=output_b_init,
                weight_normalization=weight_normalization)
            if batch_normalization:
                l_out = ly.batch_norm(l_out)
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out

            LayersPowered.__init__(self, l_out)
            initialize_tf_vars(tf.get_default_session())

    def copy(self,  trainable=True, name=None):
        args = list(self._Serializable__args)
        # -1 to account for `self` arg which is in the spec but not passed to __init__
        trainable_index = inspect.getfullargspec(self.__init__).args.index('trainable') - 1
        name_index = inspect.getfullargspec(self.__init__).args.index('name') - 1
        args[trainable_index] = trainable
        args[name_index] = name
        copy = MLP(*args)
        ly.set_all_param_values(copy.output_layer, ly.get_all_param_values(self.output_layer))
        return copy

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers

class ParamNet(LayersPowered, Serializable):
    def __init__(
            self,
            output_dim,
            initializer=tf.zeros_initializer(),
            name=None,
            input_var=None,
            input_layer=None,
            input_shape=None,
    ):

        Serializable.quick_init(self, locals())

        with tf.variable_scope(name, "ParamNet"):
            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, ) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            self._layers = [l_in]
            l_out = ly.ParamLayer(
                l_in,
                num_units=output_dim,
                param=initializer,
                name="param",
                trainable=True
                )
            self._layers.append(l_out)
            self._l_in = l_in
            self._l_out = l_out

            LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def layers(self):
        return self._layers


class ConvNetwork(LayersPowered, Serializable):
    def __init__(self,
                 input_shape,
                 output_dim,
                 conv_filters,
                 conv_filter_sizes,
                 conv_strides,
                 conv_pads,
                 hidden_sizes,
                 hidden_nonlinearity,
                 output_nonlinearity,
                 name=None,
                 hidden_w_init=ly.XavierUniformInitializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_w_init=ly.XavierUniformInitializer(),
                 output_b_init=tf.zeros_initializer(),
                 input_var=None,
                 input_layer=None,
                 batch_normalization=False,
                 weight_normalization=False):
        Serializable.quick_init(self, locals())
        """
        A network composed of several convolution layers followed by some fc
        layers.
        input_shape: (width,height,channel)
            HOWEVER, network inputs are assumed flattened. This network will
            first unflatten the inputs and then apply the standard convolutions
            and so on.
        conv_filters: a list of numbers of convolution kernel
        conv_filter_sizes: a list of sizes (int) of the convolution kernels
        conv_strides: a list of strides (int) of the conv kernels
        conv_pads: a list of pad formats (either 'SAME' or 'VALID')
        hidden_nonlinearity: a nonlinearity from tf.nn, shared by all conv and
         fc layers
        hidden_sizes: a list of numbers of hidden units for all fc layers
        """
        with tf.variable_scope(name, "ConvNetwork"):
            if input_layer is not None:
                l_in = input_layer
                l_hid = l_in
            elif len(input_shape) == 3:
                l_in = ly.InputLayer(
                    shape=(None, np.prod(input_shape)),
                    input_var=input_var,
                    name="input")
                l_hid = ly.reshape(
                    l_in, ([0], ) + input_shape, name="reshape_input")
            elif len(input_shape) == 2:
                l_in = ly.InputLayer(
                    shape=(None, np.prod(input_shape)),
                    input_var=input_var,
                    name="input")
                input_shape = (1, ) + input_shape
                l_hid = ly.reshape(
                    l_in, ([0], ) + input_shape, name="reshape_input")
            else:
                l_in = ly.InputLayer(
                    shape=(None, ) + input_shape,
                    input_var=input_var,
                    name="input")
                l_hid = l_in

            if batch_normalization:
                l_hid = ly.batch_norm(l_hid)
            for idx, conv_filter, filter_size, stride, pad in zip(
                    range(len(conv_filters)),
                    conv_filters,
                    conv_filter_sizes,
                    conv_strides,
                    conv_pads,
            ):
                l_hid = ly.Conv2DLayer(
                    l_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_hid = ly.batch_norm(l_hid)

            if output_nonlinearity == ly.spatial_expected_softmax:
                assert not hidden_sizes
                assert output_dim == conv_filters[-1] * 2
                l_hid.nonlinearity = tf.identity
                l_out = ly.SpatialExpectedSoftmaxLayer(l_hid)
            else:
                l_hid = ly.flatten(l_hid, name="conv_flatten")
                for idx, hidden_size in enumerate(hidden_sizes):
                    l_hid = ly.DenseLayer(
                        l_hid,
                        num_units=hidden_size,
                        nonlinearity=hidden_nonlinearity,
                        name="hidden_%d" % idx,
                        w=hidden_w_init,
                        b=hidden_b_init,
                        weight_normalization=weight_normalization,
                    )
                    if batch_normalization:
                        l_hid = ly.batch_norm(l_hid)
                l_out = ly.DenseLayer(
                    l_hid,
                    num_units=output_dim,
                    nonlinearity=output_nonlinearity,
                    name="output",
                    w=output_w_init,
                    b=output_b_init,
                    weight_normalization=weight_normalization,
                )
                if batch_normalization:
                    l_out = ly.batch_norm(l_out)
            self._l_in = l_in
            self._l_out = l_out
            # self._input_var = l_in.input_var

        LayersPowered.__init__(self, l_out)

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


class GRUNetwork:
    def __init__(self,
                 input_shape,
                 output_dim,
                 hidden_dim,
                 name=None,
                 hidden_nonlinearity=tf.nn.relu,
                 gru_layer_cls=ly.GRULayer,
                 output_nonlinearity=None,
                 input_var=None,
                 input_layer=None,
                 layer_args=None):
        with tf.variable_scope(name, "GRUNetwork"):
            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, None) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            l_step_input = ly.InputLayer(
                shape=(None, ) + input_shape, name="step_input")
            l_step_prev_state = ly.InputLayer(
                shape=(None, hidden_dim), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_gru = gru_layer_cls(
                l_in,
                num_units=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_init_trainable=False,
                name="gru",
                **layer_args)
            l_gru_flat = ly.ReshapeLayer(
                l_gru, shape=(-1, hidden_dim), name="gru_flat")
            l_output_flat = ly.DenseLayer(
                l_gru_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat")
            l_output = ly.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input: tf.reshape(
                    flat_output,
                    tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))
                ),
                shape_op=lambda flat_output_shape, l_input_shape: (
                    l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output")
            l_step_state = l_gru.get_step_layer(
                l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = l_step_state
            l_step_output = ly.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                w=l_output_flat.w,
                b=l_output_flat.b,
                name="step_output")

            self._l_in = l_in
            self._hid_init_param = l_gru.h0
            self._l_gru = l_gru
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim

    @property
    def hidden_dim(self):
        return self._hidden_dim

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_gru

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def state_init_param(self):
        return self._hid_init_param


class LSTMNetwork:
    def __init__(self,
                 input_shape,
                 output_dim,
                 hidden_dim,
                 name=None,
                 hidden_nonlinearity=tf.nn.relu,
                 lstm_layer_cls=ly.LSTMLayer,
                 output_nonlinearity=None,
                 input_var=None,
                 input_layer=None,
                 forget_bias=1.0,
                 use_peepholes=False,
                 layer_args=None):
        with tf.variable_scope(name, "LSTMNetwork"):
            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, None) + input_shape,
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer
            l_step_input = ly.InputLayer(
                shape=(None, ) + input_shape, name="step_input")
            # contains previous hidden and cell state
            l_step_prev_state = ly.InputLayer(
                shape=(None, hidden_dim * 2), name="step_prev_state")
            if layer_args is None:
                layer_args = dict()
            l_lstm = lstm_layer_cls(
                l_in,
                num_units=hidden_dim,
                hidden_nonlinearity=hidden_nonlinearity,
                hidden_init_trainable=False,
                name="lstm_layer",
                forget_bias=forget_bias,
                cell_init_trainable=False,
                use_peepholes=use_peepholes,
                **layer_args)
            l_lstm_flat = ly.ReshapeLayer(
                l_lstm, shape=(-1, hidden_dim), name="lstm_flat")
            l_output_flat = ly.DenseLayer(
                l_lstm_flat,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output_flat")
            l_output = ly.OpLayer(
                l_output_flat,
                op=lambda flat_output, l_input: tf.reshape(
                    flat_output,
                    tf.stack((tf.shape(l_input)[0], tf.shape(l_input)[1], -1))
                ),
                shape_op=lambda flat_output_shape, l_input_shape: (
                    l_input_shape[0], l_input_shape[1], flat_output_shape[-1]),
                extras=[l_in],
                name="output")
            l_step_state = l_lstm.get_step_layer(
                l_step_input, l_step_prev_state, name="step_state")
            l_step_hidden = ly.SliceLayer(
                l_step_state, indices=slice(hidden_dim), name="step_hidden")
            l_step_cell = ly.SliceLayer(
                l_step_state,
                indices=slice(hidden_dim, None),
                name="step_cell")
            l_step_output = ly.DenseLayer(
                l_step_hidden,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                w=l_output_flat.w,
                b=l_output_flat.b,
                name="step_output")

            self._l_in = l_in
            self._hid_init_param = l_lstm.h0
            self._cell_init_param = l_lstm.c0
            self._l_lstm = l_lstm
            self._l_out = l_output
            self._l_step_input = l_step_input
            self._l_step_prev_state = l_step_prev_state
            self._l_step_hidden = l_step_hidden
            self._l_step_cell = l_step_cell
            self._l_step_state = l_step_state
            self._l_step_output = l_step_output
            self._hidden_dim = hidden_dim

    @property
    def state_dim(self):
        return self._hidden_dim * 2

    @property
    def input_layer(self):
        return self._l_in

    @property
    def input_var(self):
        return self._l_in.input_var

    @property
    def output_layer(self):
        return self._l_out

    @property
    def recurrent_layer(self):
        return self._l_lstm

    @property
    def step_input_layer(self):
        return self._l_step_input

    @property
    def step_prev_state_layer(self):
        return self._l_step_prev_state

    @property
    def step_hidden_layer(self):
        return self._l_step_hidden

    @property
    def step_state_layer(self):
        return self._l_step_state

    @property
    def step_cell_layer(self):
        return self._l_step_cell

    @property
    def step_output_layer(self):
        return self._l_step_output

    @property
    def hid_init_param(self):
        return self._hid_init_param

    @property
    def cell_init_param(self):
        return self._cell_init_param

    @property
    def state_init_param(self):
        return tf.concat(
            axis=0, values=[self._hid_init_param, self._cell_init_param])


class ConvMergeNetwork(LayersPowered, Serializable):
    """
    This network allows the input to consist of a convolution-friendly
    component, plus a non-convolution-friendly component. These two components
    will be concatenated in the fully connected layers. There can also be a
    list of optional layers for the non-convolution-friendly component alone.


    The input to the network should be a matrix where each row is a single
    input entry, with both the aforementioned components flattened out and then
    concatenated together

    base_hidden_sizes - fc layers for the convolutional component before concatenation
    extra_hidden_sizes - fc layers for non-conv component before concatenation
    hidden_sizes - fc layers after concatenation
    """

    def __init__(self,
                 input_shape,
                 extra_input_shape,
                 output_dim,
                 hidden_sizes,
                 conv_filters,
                 conv_filter_sizes,
                 conv_strides,
                 conv_pads,
                 pool_sizes=None,
                 name=None,
                 base_hidden_sizes=None,
                 extra_hidden_sizes=None,
                 hidden_w_init=ly.XavierUniformInitializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_w_init=ly.XavierUniformInitializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 input_var=None,
                 input_layer=None):
        Serializable.quick_init(self, locals())

        if extra_hidden_sizes is None:
            extra_hidden_sizes = []
        if base_hidden_sizes is None:
            base_hidden_sizes = []
        if pool_sizes is None:
            pool_sizes = (None, ) * len(conv_filters)

        with tf.variable_scope(name, "ConvMergeNetwork"):

            input_flat_dim = np.prod(input_shape)
            extra_input_flat_dim = np.prod(extra_input_shape)
            total_input_flat_dim = input_flat_dim + extra_input_flat_dim

            if input_layer is None:
                l_in = ly.InputLayer(
                    shape=(None, total_input_flat_dim),
                    input_var=input_var,
                    name="input")
            else:
                l_in = input_layer

            l_conv_in = ly.reshape(
                ly.SliceLayer(
                    l_in, indices=slice(input_flat_dim), name="conv_slice"),
                ([0], ) + input_shape,
                name="conv_reshaped")
            l_extra_in = ly.reshape(
                ly.SliceLayer(
                    l_in,
                    indices=slice(input_flat_dim, None),
                    name="extra_slice"), ([0], ) + extra_input_shape,
                name="extra_reshaped")

            l_conv_hid = l_conv_in
            for idx, conv_filter, filter_size, stride, pad, pool_size in zip(
                    range(len(conv_filters)), conv_filters, conv_filter_sizes,
                    conv_strides, conv_pads, pool_sizes):
                l_conv_hid = ly.Conv2DLayer(
                    l_conv_hid,
                    num_filters=conv_filter,
                    filter_size=filter_size,
                    stride=(stride, stride),
                    pad=pad,
                    nonlinearity=hidden_nonlinearity,
                    name="conv_hidden_%d" % idx,
                )
                if pool_size is not None:
                    l_conv_hid = ly.Pool2DLayer(
                        l_conv_hid,
                        pool_size,
                        name="max_pool_%d" % idx,
                    )

            for idx, hidden_size in enumerate(base_hidden_sizes):
                l_conv_hid = ly.DenseLayer(
                    l_conv_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="base_hidden_%d" % idx,
                    w=hidden_w_init,
                    b=hidden_b_init,
                )
                
            l_extra_hid = l_extra_in
            for idx, hidden_size in enumerate(extra_hidden_sizes):
                l_extra_hid = ly.DenseLayer(
                    l_extra_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="extra_hidden_%d" % idx,
                    w=hidden_w_init,
                    b=hidden_b_init,
                )


            l_joint_hid = ly.concat(
                [ly.flatten(l_conv_hid, name="conv_hidden_flat"), l_extra_hid],
                name="joint_hidden")

            for idx, hidden_size in enumerate(hidden_sizes):
                l_joint_hid = ly.DenseLayer(
                    l_joint_hid,
                    num_units=hidden_size,
                    nonlinearity=hidden_nonlinearity,
                    name="joint_hidden_%d" % idx,
                    w=hidden_w_init,
                    b=hidden_b_init,
                )
            l_out = ly.DenseLayer(
                l_joint_hid,
                num_units=output_dim,
                nonlinearity=output_nonlinearity,
                name="output",
                w=output_w_init,
                b=output_b_init,
            )
            self._l_in = l_in
            self._l_out = l_out

            LayersPowered.__init__(self, [l_out], input_layers=[l_in])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var


class MDN(LayersPowered, Serializable):
   
    def __init__(self,
                 body,
                 M,
                 dim=1,
                 mean_nonlinearity=tf.identity,
                 std_nonlinearity=tf.nn.sigmoid,
                 weight_nonlinearity=tf.nn.softmax,
                 name=None,
                 output_w_init=ly.XavierUniformInitializer(),
                 output_b_init=tf.zeros_initializer()):
        Serializable.quick_init(self, locals())
        """
        This network represents a mixture of Gaussians.
        The outputs are the means and log_stds of M Gaussian distributions as well
        as the mixing weights of those distributions
    
        It takes a LayersPowered network as input and replaces the final layer
        :param body: a LayersPowered network. The final layer will be replaced
        with a FC layer that outputs the GMM statistics
        :param M: the number of mixture elements
        :param d: the dimensionality of each Gaussian
        """

        self.M = M
        self.dim = dim
        with tf.variable_scope(name, "MDN"):
            l_hid = body._l_out.input_layer
            self._l_mean = ly.DenseLayer(
                l_hid,
                num_units=M*dim,
                nonlinearity=mean_nonlinearity,
                name="mixture_means",
                w=output_w_init,
                b=output_b_init,
            )
            self._l_mean = ly.reshape(self._l_mean, ([0], M, dim))
            self._l_std = ly.DenseLayer(
                l_hid,
                num_units=M*dim,
                nonlinearity=std_nonlinearity,
                name="mixture_stds",
                w=output_w_init,
                b=output_b_init,
            )
            self._l_std = ly.reshape(self._l_std, ([0], M, dim))
            self._l_w = ly.DenseLayer(
                l_hid,
                num_units=M,
                nonlinearity=tf.tanh,
                name="mixture_weights",
                w=output_w_init,
                b=output_b_init,
            )
            self._l_w = ly.NonlinearityLayer(self._l_w, weight_nonlinearity)
            self._l_in = body._l_in
            self._l_out = [self._l_mean, self._l_std, self._l_w]

            LayersPowered.__init__(self, self._l_out, input_layers=[self._l_in])

    @property
    def input_layer(self):
        return self._l_in

    @property
    def output_layer(self):
        return self._l_out

    @property
    def input_var(self):
        return self._l_in.input_var
