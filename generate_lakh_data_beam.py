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
r"""Dataset generation."""

import pickle

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
from apache_beam.metrics import Metrics
from magenta.models.music_vae import TrainedModel
import note_seq

import config
from utils import song_utils
from utils import myns_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'pipeline_options', '--runner=DirectRunner',
    'Command line flags to use in constructing the Beam pipeline options.')

# Model
flags.DEFINE_string('model', 'melody-2-big', 'Model configuration.')
flags.DEFINE_string('checkpoint', '/home/iid/wxy/symbolic-music-diffusion/checkpoints/musicvae_ckpt/cat-mel_2bar_big.tar',
                    'Model checkpoint.')

# Data transformation
flags.DEFINE_enum('mode', 'melody', ['melody', 'multitrack'],
                  'Data generation mode.')
flags.DEFINE_enum('split_mode','pitch', ['part','inst','pitch'],
                  'Music part split mode.')
flags.DEFINE_enum('split_typename1', 'p2',['bach_soprano','bach_alto','piano','strings','melinst','bass','guitar','soprano','p1','p2'],
                  'Music part to be extracted.')
flags.DEFINE_enum('split_typename2', 'p1',['bach_soprano','bach_alto','piano','strings','melinst','bass','guitar','soprano','p1','p2'],
                  'Music part to be extracted.')
flags.DEFINE_string('input', '/home/iid/wxy/forked-symbolic-music-diffusion/datasets/lmd_clean/lmdc_eval.tfrecord', 'Path to tfrecord files.')
flags.DEFINE_string('output', '/home/iid/wxy/forked-symbolic-music-diffusion/datasets/lmd_clean/encoded/eval_y_p2', 'Output path.')

split_info_dict = {}
split_info_dict['bach_soprano'] = [1]
split_info_dict['bach_alto'] = [2]
split_info_dict['piano'] = [0,1,2,3,4,5,6,7]
split_info_dict['strings'] = [40,41,42,43,44,45,46,47]
split_info_dict['melinst'] = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
split_info_dict['bass'] = [32,33,34,35,36,37,38,39]
split_info_dict['guitar'] = [24,25,26,27,28,29,30,31]
split_info_dict['p1'] = range(-2,61)
split_info_dict['p2'] = range(61,128)

class EncodeSong(beam.DoFn):
  """Encode song into MusicVAE embeddings."""

  def setup(self):
    logging.info('Loading pre-trained model %s', FLAGS.model)
    self.model_config = config.MUSIC_VAE_CONFIG[FLAGS.model]
    self.model = TrainedModel(self.model_config,
                              batch_size=1,
                              checkpoint_dir_or_path=FLAGS.checkpoint)

  def process(self, ns):
    logging.info('Processing %s::%s (%f)', ns.id, ns.filename, ns.total_time)
    if ns.total_time > 60 * 60:
      logging.info('Skipping notesequence with >1 hour duration')
      Metrics.counter('EncodeSong', 'skipped_long_song').inc()
      return

    Metrics.counter('EncodeSong', 'encoding_song').inc()

    if FLAGS.mode == 'melody':
      chunk_length = 2
      #melodies = song_utils.extract_melodies(ns)
      melodies = myns_utils.extract_melodies(ns,keep_longest_split=False, 
                                             mode=FLAGS.split_mode, mode_related_info=split_info_dict[FLAGS.split_typename1])
      if not melodies:
        Metrics.counter('EncodeSong', 'extracted_no_melodies').inc()
        return
      Metrics.counter('EncodeSong', 'extracted_melody').inc(len(melodies))
      songs = [
          song_utils.Song(melody, self.model_config.data_converter,
                          chunk_length) for melody in melodies
      ]
      encoding_matrices = song_utils.encode_songs(self.model, songs)

      melodies = myns_utils.extract_melodies(ns,keep_longest_split=False, 
                                             mode=FLAGS.split_mode, mode_related_info=split_info_dict[FLAGS.split_typename2])
      if not melodies:
        Metrics.counter('EncodeSong', 'extracted_no_melodies').inc()
        return
      Metrics.counter('EncodeSong', 'extracted_melody').inc(len(melodies))
      songs = [
          song_utils.Song(melody, self.model_config.data_converter,
                          chunk_length) for melody in melodies
      ]
      encoding_matrices2 = song_utils.encode_songs(self.model, songs)
    elif FLAGS.mode == 'multitrack':
      chunk_length = 1
      song = song_utils.Song(ns,
                             self.model_config.data_converter,
                             chunk_length,
                             multitrack=True)
      encoding_matrices = song_utils.encode_songs(self.model, [song])
    else:
      raise ValueError(f'Unsupported mode: {FLAGS.mode}')

    for idx, matrix in enumerate(encoding_matrices):
      assert matrix.shape[0] == 3 and matrix.shape[-1] == 512
      if matrix.shape[1] == 0:
        Metrics.counter('EncodeSong', 'skipped_matrix').inc()
        continue
      if matrix.shape != encoding_matrices2[idx].shape:
        continue
      logging.info('matrix shape: {}'.format(matrix.shape))
      Metrics.counter('EncodeSong', 'encoded_matrix').inc()
      yield pickle.dumps(matrix)


def main(argv):
  del argv  # unused

  pipeline_options = beam.options.pipeline_options.PipelineOptions(
      FLAGS.pipeline_options.split(','))

  with beam.Pipeline(options=pipeline_options) as p:
    p |= 'tfrecord_list' >> beam.Create([FLAGS.input])
    p |= 'read_tfrecord' >> beam.io.tfrecordio.ReadAllFromTFRecord(
        coder=beam.coders.ProtoCoder(note_seq.NoteSequence))
    p |= 'shuffle_input' >> beam.Reshuffle()
    p |= 'encode_song' >> beam.ParDo(EncodeSong())
    p |= 'shuffle_output' >> beam.Reshuffle()
    p |= 'write' >> beam.io.WriteToTFRecord(FLAGS.output)


if __name__ == '__main__':
  app.run(main)
