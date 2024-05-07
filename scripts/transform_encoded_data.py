# Copyright 2021 The Magenta Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Transforms an encoded song dataset into an appropriate format
for a model.
"""
import glob
import os
import pickle
import sys

from absl import app
from absl import flags
from absl import logging
from functools import reduce

import numpy as np
import tensorflow as tf

sys.path.append("{}/../".format(os.path.dirname(os.path.abspath(__file__))))
import utils.data_utils as data_utils

FLAGS = flags.FLAGS

flags.DEFINE_boolean('toy_data', False, 'Create a toy dataset.')
flags.DEFINE_string('encoded_data', '/home/iid/wxy/forked-symbolic-music-diffusion/datasets/bcdf/encoded',
                    'Path to encoded data TFRecord directory.')
flags.DEFINE_string('output_path', '/home/iid/wxy/forked-symbolic-music-diffusion/datasets/bcdf/ncsn_input', 'Output directory.')
flags.DEFINE_integer('shard_size', 2**17, 'Number of vectors per shard.')
flags.DEFINE_enum('output_format', 'tfrecord', ['tfrecord', 'pkl'],
                  'Shard file type.')

flags.DEFINE_enum('mode', 'sequences', ['flatten', 'sequences', 'decoded'],
                  'Transformation mode.')
flags.DEFINE_boolean('remove_zeros', True, 'Remove zero vectors.')
flags.DEFINE_integer('context_length', 32,
                     'The length of the context window in a sequence.')
flags.DEFINE_integer('stride', 1, 'The stride used for generating sequences.')
flags.DEFINE_integer('max_songs', None,
                     'The maximum number of songs to process.')
flags.DEFINE_integer('max_examples', None,
                     'The maximum number of examples to process.')


def _bytes_feature(value):
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy()
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(values):
  return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def _int_feature(values):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _serialize(writer, input_tensor, target_tensor=None):
  assert writer is not None
  prod = lambda a: reduce(lambda x, y: x * y, a)
  input_shape = input_tensor.shape
  inputs = input_tensor.reshape(prod(input_shape),)

  if FLAGS.mode == 'decoded':
    sequence = tf.io.serialize_tensor(input_tensor)
    features = _bytes_feature(sequence)
  else:
    features = _float_feature(inputs)

  features = {'inputs': features, 'input_shape': _int_feature(input_shape)}

  if target_tensor is not None:
    print("target is not none")
    target_shape = target_tensor.shape
    targets = target_tensor.reshape(prod(target_shape),)
    features['targets'] = _float_feature(targets)
    features['target_shape'] = _int_feature(target_shape)
  #print("feature map:",features)
  tf_example = tf.train.Example(features=tf.train.Features(feature=features))
  writer.write(tf_example.SerializeToString())


def _serialize_tf_shard(shard, output_path):
  with tf.io.TFRecordWriter(os.path.expanduser(output_path)) as writer:
    if FLAGS.mode == 'sequences':
      for context, target in zip(*shard):
        _serialize(writer, context, target_tensor=target)
    elif FLAGS.mode == 'flatten' or FLAGS.mode == 'decoded':
      for example in shard:
        _serialize(writer, example)
  logging.info('Saved to %s', output_path)


def save_shard(contexts, targets, output_path):
  if FLAGS.mode == 'flatten' or FLAGS.mode == 'decoded':
    shard = targets[:FLAGS.shard_size]

    shard_type = np.bool if FLAGS.mode == 'decoded' else np.float32
    shard = np.stack(shard).astype(shard_type)

    targets = targets[FLAGS.shard_size:]
  elif FLAGS.mode == 'sequences':
    context_shard = contexts[:FLAGS.shard_size]
    target_shard = targets[:FLAGS.shard_size]
    context_shard = np.stack(context_shard).astype(np.float32)
    target_shard = np.stack(target_shard).astype(np.float32)
    shard = (context_shard, target_shard)

    contexts = contexts[FLAGS.shard_size:]
    targets = targets[FLAGS.shard_size:]
    
  output_path += '.' + FLAGS.output_format

  # Serialize shard
  if FLAGS.output_format == 'pkl':
    data_utils.save(shard, output_path)
  elif FLAGS.output_format == 'tfrecord':
    _serialize_tf_shard(shard, output_path)

  return contexts, targets


def toy_distribution_fn(batch_size=512):
  """Samples from a 0.2 * N(-5, 1) + 0.8 * N(5, 1)."""

  c1 = (np.random.randn(batch_size, 2) + 5)
  c2 = (np.random.randn(batch_size, 2) + -5)
  mask = np.random.uniform(size=batch_size) < 0.8
  mask = mask[:, np.newaxis]
  mixture = mask * c1 + (1 - mask) * c2
  return mixture


def toy_sequence_distribution_fn(trajectory_length=10, batch_size=512):
  c1 = 0.01 * np.random.randn(batch_size, 2) + 5
  c2 = 0.01 * np.random.randn(batch_size, 2) + -5
  mask = np.random.uniform(size=batch_size) < 0.8
  mask = mask[:, np.newaxis]
  center = mask * c1 + (1 - mask) * c2
  step_size = 0.1 * np.random.randn(batch_size, 2)
  deltas = np.expand_dims(step_size, 1).repeat(
      trajectory_length, axis=1) * np.arange(trajectory_length).reshape(
          trajectory_length, 1)
  center = np.expand_dims(center, 1).repeat(trajectory_length, axis=1)
  return center + deltas


def main(argv):
  del argv  # unused
  print("enc path",FLAGS.encoded_data)
  if FLAGS.mode == 'decoded':
    train_glob = f'{FLAGS.encoded_data}/decoded-train.tfrecord-*'
    eval_glob = f'{FLAGS.encoded_data}/decoded-eval.tfrecord-*'
  else:
    trainx_glob = f'{FLAGS.encoded_data}/*'
    evalx_glob = f'{FLAGS.encoded_data}/*'
    trainy_glob = f'{FLAGS.encoded_data}/*'
    evaly_glob = f'{FLAGS.encoded_data}/*'
  print(os.path.expanduser(trainx_glob))
  trainx_files = glob.glob(os.path.expanduser(trainx_glob))
  evalx_files = glob.glob(os.path.expanduser(evalx_glob))
  trainy_files = glob.glob(os.path.expanduser(trainy_glob))
  evaly_files = glob.glob(os.path.expanduser(evaly_glob))
  print("trainfile len",len(trainx_files))

  tensor_shape = [tf.float64]
  trainx_dataset = tf.data.TFRecordDataset(
      trainx_files).map(lambda x: tf.py_function(
          lambda binary: pickle.loads(binary.numpy()), [x], tensor_shape),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  evalx_dataset = tf.data.TFRecordDataset(
      evalx_files).map(lambda x: tf.py_function(
          lambda binary: pickle.loads(binary.numpy()), [x], tensor_shape),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  trainy_dataset = tf.data.TFRecordDataset(
      trainy_files).map(lambda x: tf.py_function(
          lambda binary: pickle.loads(binary.numpy()), [x], tensor_shape),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
  evaly_dataset = tf.data.TFRecordDataset(
      evaly_files).map(lambda x: tf.py_function(
          lambda binary: pickle.loads(binary.numpy()), [x], tensor_shape),
                      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  ctx_window = FLAGS.context_length
  stride = FLAGS.stride

  for ds, split in [(trainx_dataset, 'trainx'), (evalx_dataset, 'evalx'), 
                    (trainy_dataset, 'trainy'), (evaly_dataset, 'evaly')]:
    if FLAGS.max_songs is not None:
      ds = ds.take(FLAGS.max_songs)

    output_fp = '{}/{}-{:04d}'
    contexts, targets = [], []
    count = 0
    discard = 0
    example_count, should_terminate = 0, False
    for song_data in ds.as_numpy_iterator():
      song_embeddings = song_data[0]

      if FLAGS.mode != 'decoded':
        assert song_embeddings.ndim == 3 and song_embeddings.shape[0] == 3

        # Use the full VAE embedding
        song = song_embeddings[0]
        #print("song shape:",song.shape)

      else:
        song = song_data[0]
        if song.shape[0] < 896:
          discard += 1
          continue

        pad_len = 1024 - song.shape[0]
        padding = np.zeros((pad_len, song.shape[-1]))
        padding[:, 0] = 1.
        song = np.concatenate((song, padding))
        assert song.shape[0] == 1024 and song.ndim == 2

      if FLAGS.mode == 'decoded':
        example_count += 1
        targets.append(song)

      if FLAGS.toy_data:
        song = toy_distribution_fn(batch_size=len(song))

      if FLAGS.mode == 'flatten':
        for vec in song:
          if FLAGS.remove_zeros and np.linalg.norm(vec) < 1e-6:
            continue
          if FLAGS.max_examples is not None and example_count >= FLAGS.max_examples:
            should_terminate = True
            break
          example_count += 1
          targets.append(vec)
      elif FLAGS.mode == 'sequences':
        for i in range(0, len(song) - ctx_window, stride):
          logging.info(f'len(song) is {len(song)}, i is {i}.')
          context = song[i:i + ctx_window]
          if FLAGS.remove_zeros and np.where(
              np.linalg.norm(context, axis=1) < 1e-6)[0].any():
            continue
          if FLAGS.max_examples is not None and example_count >= FLAGS.max_examples:
            should_terminate = True
            break
          example_count += 1
          contexts.append(context)
          targets.append(song[i + ctx_window])
        #print("context list length:", len(contexts))

      if len(targets) >= FLAGS.shard_size:
        contexts, targets = save_shard(
            contexts, targets, output_fp.format(FLAGS.output_path, split,
                                                count))
        count += 1

      if should_terminate:
        break

    logging.info(f'Discarded {discard} invalid sequences.')
    logging.info(f'len(targets) is {len(targets)}.')
    if len(targets) > 0:
      save_shard(contexts, targets,
                 output_fp.format(FLAGS.output_path, split, count))


if __name__ == '__main__':
  app.run(main)
