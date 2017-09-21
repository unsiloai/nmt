import tensorflow as tf
import functools
from tensorflow.python.util import nest
from tensorflow.python.framework import ops
from tensorflow.contrib.rnn import LSTMStateTuple
from tensorflow.contrib.rnn import RNNCell

# from seq2seq.data.parallel_data_provider import make_parallel_data_provider

__all__ = ["GatedFeedbackMultiRNNCell"]


def lazy_property(function):
  attribute = '_' + function.__name__

  @property
  @functools.wraps(function)
  def wrapper(self):
    if not hasattr(self, attribute):
      setattr(self, attribute, function(self))
    return getattr(self, attribute)

  return wrapper


class GatedFeedbackMultiRNNCell(RNNCell):
  """RNN cell composed sequentially of multiple simple cells."""

  def __init__(self, cells):
    """Create a RNN cell composed sequentially of a number of RNNCells.
    Args:
      cells: list of RNNCells that will be composed in this order.
    Raises:
      ValueError: if cells is empty (not allowed), or at least one of the cells
        returns a state tuple but the flag `state_is_tuple` is `False`.
    """
    if not cells:
      raise ValueError("Must specify at least one cell for MultiRNNCell.")
    if not nest.is_sequence(cells):
      raise TypeError(
        "cells must be a list or tuple, but saw: %s." % cells)

    self._cells = cells

  @property
  def state_size(self):
    return tuple(cell.state_size for cell in self._cells)

  def zero_state(self, batch_size, dtype):
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      states = [cell.zero_state(batch_size=batch_size, dtype=dtype) for cell in self._cells]
      return tuple(states)

  def get_h_from_state(self, state):
    h_states = []
    for idx, cell in enumerate(self._cells):
      h = cell.get_h_from_state(state[idx])
      h_states.append(h)
    return tuple(h_states)

  def get_c_from_state(self, state):
    c_states = []
    for idx, cell in enumerate(self._cells):
      c = cell.get_c_from_state(state[idx])
      c_states.append(c)
    return tuple(c_states)

  def recreate_state(self, c, h, old_state):
    states = []
    for idx, cell in enumerate(self._cells):
      states.append(cell.recreate_state(c[idx], h[idx], old_state[idx]))
    return tuple(states)

  @property
  def output_size(self):
    return self._cells[-1].output_size

  def __call__(self, inputs, state, scope=None):
    """Run this multi-layer cell on inputs, starting from state."""
    with tf.variable_scope(scope or "gf_multi_rnn_cell"):
      cur_inp = inputs
      new_states = []

      global_activation_states = [cell.get_h_from_state(layer_state) for cell, layer_state in zip(self._cells, state)]
      global_activation_states_concat = tf.concat(global_activation_states, -1)

      print("global_activation_states_concat", global_activation_states_concat)

      # Calculate each layer
      for i, cell in enumerate(self._cells):
        with tf.variable_scope("cell_%d" % i):
          if not nest.is_sequence(state):
            raise ValueError(
              "Expected state to be a tuple of length %d, but received: %s"
              % (len(self.state_size), state))
          cur_state = state[i]

          # try:
          cur_inp, new_state = cell(
            cur_inp,
            cur_state,
            scope=None,
            global_activation_states=global_activation_states,
            global_activation_states_concat=global_activation_states_concat,
            layer_count=len(self._cells)
          )
          # except ValueError as err:
          #     raise ValueError(
          #         "RNNCell was wrapped with GatedFeedbackMultiRNNCell, but does not support it: ")

          new_states.append(new_state)
      new_states = tuple(new_states)
    return cur_inp, new_states