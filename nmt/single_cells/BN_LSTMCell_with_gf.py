# Copyright (C) 2016 by Akira TAMAMORI
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <http://www.gnu.org/licenses/>.

# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Notice:
# This file is tested on TensorFlow v0.10.0 only.

# Commentary:
# TODO: implemation of another initializer for LSTM

import numpy as np
import tensorflow as tf

from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple

__all__ = ["BN_LSTMCell"]


# Thanks to 'initializers_enhanced.py' of Project RNN Enhancement:
# https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow/blob/master/rnn_enhancement/initializers_enhanced.py
def orthogonal_initializer(scale=1.0):
  def _initializer(shape, dtype=tf.float32, partition_info=None):
    if partition_info is not None:
      ValueError("Do not know what to do with partition_info in BN_LSTMCell")
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return tf.constant(scale * q[:shape[0], :shape[1]], dtype=dtype)

  return _initializer


# Thanks to https://github.com/OlavHN/bnlstm
def batch_norm(inputs, name_scope, is_training, time_step, statistics_length, epsilon=1e-3, decay=0.99):
  with tf.variable_scope(name_scope):
    size = inputs.get_shape().as_list()[1]

    scale = tf.get_variable(
      'scale', [size], initializer=tf.constant_initializer(0.1))
    offset = tf.get_variable('offset', [size])

    population_mean = tf.get_variable(
      'population_mean', [statistics_length, size],
      initializer=tf.zeros_initializer(), trainable=False)
    population_var = tf.get_variable(
      'population_var', [statistics_length, size],
      initializer=tf.ones_initializer(), trainable=False)
    batch_mean, batch_var = tf.nn.moments(inputs, [0])

    # The following part is based on the implementation of :
    # https://github.com/cooijmanstim/recurrent-batch-normalization
    def update_pop_mean():
      update_assert_mean = tf.Assert(tf.less(time_step, statistics_length), [time_step])
      with tf.control_dependencies([update_assert_mean]):
        return tf.scatter_nd_update(
          population_mean,
          [[time_step]],
          [population_mean[time_step] * decay + batch_mean * (1 - decay)])

    def update_pop_var():
      update_assert_var = tf.Assert(tf.less(time_step, statistics_length), [time_step])
      with tf.control_dependencies([update_assert_var]):
        return tf.scatter_nd_update(
          population_var,
          [[time_step]],
          [population_var[time_step] * decay + batch_var * (1 - decay)])

    train_mean_op = tf.cond(tf.less(time_step, statistics_length),
                            update_pop_mean,
                            lambda: tf.zeros((statistics_length, size)))

    train_var_op = tf.cond(tf.less(time_step, statistics_length),
                           update_pop_var,
                           lambda: tf.zeros((statistics_length, size)))

    capped_step = tf.reduce_min((time_step, statistics_length - 1))

    if is_training is True:
      with tf.control_dependencies([train_mean_op, train_var_op]):
        return tf.nn.batch_normalization(
          inputs, batch_mean, batch_var, offset, scale, epsilon)
    else:
      return tf.nn.batch_normalization(
        inputs, population_mean[capped_step], population_var[capped_step], offset, scale, epsilon)


class BN_LSTMCell(RNNCell):
  """LSTM cell with Recurrent Batch Normalization.

  This implementation is based on:
       http://arxiv.org/abs/1603.09025

  This implementation is also based on:
       https://github.com/OlavHN/bnlstm
       https://github.com/nicolas-ivanov/Seq2Seq_Upgrade_TensorFlow

  """

  def __init__(self, num_units, is_training, statistics_length,
               use_peepholes=False,
               cell_clip=None,
               initializer=orthogonal_initializer(),
               num_proj=None,
               proj_clip=None,
               forget_bias=1.0,
               use_batch_norm_h=True,
               use_batch_norm_x=True,
               use_batch_norm_c=True,
               # drop_dependency_on_h=False, // TODO: Implement training RNNs as fast as CNNs Tao Lei et. al
               activation=tf.tanh):
    """Initialize the parameters for an LSTM cell.
    Args:
      num_units: int, The number of units in the LSTM cell.
      is_training: bool, set True when training.
      statistics_length: int, How many timesteps to keep population mean and variance length
      use_peepholes: bool, set True to enable diagonal/peephole
        connections.
      cell_clip: (optional) A float value, if provided the cell state
        is clipped by this value prior to the cell output activation.
      initializer: (optional) The initializer to use for the weight
        matrices.
      num_proj: (optional) int, The output dimensionality for
        the projection matrices.  If None, no projection is performed.
      forget_bias: Biases of the forget gate are initialized by default
        to 1 in order to reduce the scale of forgetting at the beginning of
        the training.
      activation: Activation function of the inner states.
    """

    self.num_units = num_units
    self.is_training = is_training
    self.use_peepholes = use_peepholes
    self.cell_clip = cell_clip
    self.num_proj = num_proj
    self.proj_clip = proj_clip
    self.initializer = initializer
    self.forget_bias = forget_bias
    self._use_batch_norm_h = use_batch_norm_h
    self._use_batch_norm_x = use_batch_norm_x
    self._use_batch_norm_c = use_batch_norm_c
    self.activation = activation
    self.statistics_length = statistics_length

    if num_proj:
      self._state_size = (LSTMStateTuple(num_units, num_units), 1)
      self._output_size = num_proj
    else:
      self._state_size = (LSTMStateTuple(num_units, num_units), 1)
      self._output_size = num_units

  @property
  def state_size(self):
    return self._state_size

  @property
  def output_size(self):
    return self._output_size

  # @classmethod
  # def get_h_from_state(cls, state):
  #     ((c, h), time_step) = state
  #     return h

  def get_h_from_state(self, state):
    ((c, h), time_step) = state
    return h

  def get_c_from_state(self, state):
    ((c, h), time_step) = state
    return c

  def recreate_state(self, c, h, old_state):
    return ((c, h), old_state[1])

  def __call__(self, inputs, state, scope=None, global_activation_states=None, global_activation_states_concat=None,
               layer_count=None):

    num_proj = self.num_units if self.num_proj is None else self.num_proj

    ((c_prev, h_prev), time_step) = state

    # state is (LSTMStateTuple, time_step). First dimension of tensors in the state tuple must match the batch
    # dimension, therfore time_step is a [batch_size, 1]. The time step is the same for all sequences, so we just
    # pick the time step for the first sequence
    time_step_flat = tf.cast(tf.squeeze(time_step[0]), tf.int32)
    dtype = inputs.dtype
    input_size = inputs.get_shape().with_rank(2)[1]

    with tf.variable_scope(scope or type(self).__name__):
      if input_size.value is None:
        raise ValueError(
          "Could not infer input size from inputs.get_shape()[-1]")

      number_of_gates = 4

      if global_activation_states is not None:
        number_of_gates = 3

      W_xh = tf.get_variable(
        'W_xh',
        [input_size, number_of_gates * self.num_units],
        initializer=self.initializer)
      xh = tf.matmul(inputs, W_xh)

      if self._use_batch_norm_x:
        bn_xh = batch_norm(xh, 'xh', self.is_training, time_step_flat, self.statistics_length)
      else:
        bn_xh = xh

      W_hh = tf.get_variable(
        'W_hh',
        [self.num_units, number_of_gates * self.num_units],
        initializer=self.initializer)
      hh = tf.matmul(h_prev, W_hh)

      if self._use_batch_norm_h:
        bn_hh = batch_norm(hh, 'hh', self.is_training, time_step_flat, self.statistics_length)
      else:
        bn_hh = hh

      bias = tf.get_variable('B', [number_of_gates * self.num_units])

      # i:input gate, j:new input, f:forget gate, o:output gate
      lstm_matrix = tf.nn.bias_add(tf.add(bn_xh, bn_hh), bias)

      if global_activation_states is not None:
        i, f, o = tf.split(value=lstm_matrix, num_or_size_splits=number_of_gates, axis=1)

        W_j = tf.get_variable(
          'W_j',
          [input_size, self.num_units],
          initializer=self.initializer)
        xj = tf.matmul(inputs, W_j)

        bias_j = tf.get_variable('B_j', [self.num_units])

        input_size = inputs.get_shape()[-1]
        global_state_size = layer_count * self.num_units  # tf.shape(global_activation_states_concat)[-1]

        def calculate_uj_for_one_layer(counter):

          W_g = tf.get_variable(
            'W_g_%d' % counter,
            [input_size, 1],
            initializer=tf.contrib.layers.xavier_initializer())

          U_g = tf.get_variable(
            'U_g_%d' % counter,
            [global_state_size, 1],
            initializer=tf.contrib.layers.xavier_initializer())

          g_forward_part = tf.matmul(inputs, W_g)
          g_recurrent_part = tf.matmul(global_activation_states_concat, U_g)

          g = tf.sigmoid(tf.add(g_forward_part, g_recurrent_part))

          U = tf.get_variable(
            'U_%d' % counter,
            [self.num_units, self.num_units],
            initializer=self.initializer)

          cur_state = tf.matmul(global_activation_states[counter], U)

          u = tf.multiply(g, cur_state)

          return u

        u = calculate_uj_for_one_layer(0)

        # Calculate Global Reset Gates
        for counter in range(1, layer_count):
          u = tf.add(u, calculate_uj_for_one_layer(counter))

        # Finally complete GF candidate value (j) (Before activation)
        j = xj + u + bias_j

      else:
        i, j, f, o = tf.split(value=lstm_matrix, num_or_size_splits=number_of_gates, axis=1)

      # Diagonal connections
      if self.use_peepholes:
        w_f_diag = tf.get_variable(
          "W_F_diag", shape=[self.num_units], dtype=dtype)
        w_i_diag = tf.get_variable(
          "W_I_diag", shape=[self.num_units], dtype=dtype)
        w_o_diag = tf.get_variable(
          "W_O_diag", shape=[self.num_units], dtype=dtype)

      if self.use_peepholes:
        c = c_prev * tf.sigmoid(f + self.forget_bias +
                                w_f_diag * c_prev) + \
            tf.sigmoid(i + w_i_diag * c_prev) * self.activation(j)
      else:
        c = c_prev * tf.sigmoid(f + self.forget_bias) + \
            tf.sigmoid(i) * self.activation(j)

      if self.cell_clip is not None:
        c = tf.clip_by_value(c, -self.cell_clip, self.cell_clip)

      if self._use_batch_norm_c:
        bn_c = batch_norm(c, 'cell', self.is_training, time_step_flat, self.statistics_length)
      else:
        bn_c = c

      if self.use_peepholes:
        h = tf.sigmoid(o + w_o_diag * c) * self.activation(bn_c)
      else:
        h = tf.sigmoid(o) * self.activation(bn_c)

      if self.num_proj is not None:
        w_proj = tf.get_variable(
          "W_P", [self.num_units, num_proj], dtype=dtype)

        h_proj = tf.matmul(h, w_proj)
        if self.proj_clip is not None:
          h_proj = tf.clip_by_value(h_proj, -self.proj_clip, self.proj_clip)
      else:
        h_proj = h

      return h_proj, (LSTMStateTuple(c, h), time_step + 1)

  def zero_state(self, batch_size, dtype):
    return super(BN_LSTMCell, self).zero_state(batch_size, dtype)
