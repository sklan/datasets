# coding=utf-8
# Copyright 2019 The TensorFlow Datasets Authors.
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

"""Corrupted Cifar10 Dataset.

Cifar10Corrupted is a dataset generated by adding 15 common corruptions to the
test images in the Cifar10 dataset. This dataset wraps the corrupted Cifar10
test images uploaded by the original authors.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from tensorflow_datasets.core import api_utils
import tensorflow_datasets.public_api as tfds

_DESCRIPTION = """\
Cifar10Corrupted is a dataset generated by adding 15 common corruptions to the
test images in the Cifar10 dataset. This dataset wraps the corrupted Cifar10
test images uploaded by the original authors.
"""

_CITATION = """\
@inproceedings{
  hendrycks2018benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HJz6tiCqYm},
}
"""

_CIFAR_IMAGE_SIZE = (32, 32, 3)
_CIFAR_CLASSES = 10
_DOWNLOAD_URL = 'https://zenodo.org/record/2535967/files/CIFAR-10-C.tar'
_CORRUPTIONS_TO_FILENAMES = {
    'gaussian_noise': 'gaussian_noise.npy',
    'shot_noise': 'shot_noise.npy',
    'impulse_noise': 'impulse_noise.npy',
    'defocus_blur': 'defocus_blur.npy',
    'frosted_glass_blur': 'glass_blur.npy',
    'motion_blur': 'motion_blur.npy',
    'zoom_blur': 'zoom_blur.npy',
    'snow': 'snow.npy',
    'frost': 'frost.npy',
    'fog': 'fog.npy',
    'brightness': 'brightness.npy',
    'contrast': 'contrast.npy',
    'elastic': 'elastic_transform.npy',
    'pixelate': 'pixelate.npy',
    'jpeg_compression': 'jpeg_compression.npy',
}
_CORRUPTIONS, _FILENAMES = zip(*sorted(_CORRUPTIONS_TO_FILENAMES.items()))
_DIRNAME = 'CIFAR-10-C'
_LABELS_FILENAME = 'labels.npy'


class Cifar10CorruptedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Cifar10Corrupted."""

  @api_utils.disallow_positional_args
  def __init__(self, corruption_type, severity, **kwargs):
    """Constructor.

    Args:
      corruption_type: string, must be one of the items in _CORRUPTIONS.
      severity: integer, bewteen 1 and 5.
      **kwargs: keyword arguments forwarded to super.
    """
    super(Cifar10CorruptedConfig, self).__init__(**kwargs)
    self.corruption = corruption_type
    self.severity = severity


def _make_builder_configs():
  """Construct a list of BuilderConfigs.

  Construct a list of 75 Cifar10CorruptedConfig objects, corresponding to
  the 15 corruption types and 5 severities.

  Returns:
    A list of 75 Cifar10CorruptedConfig objects.
  """
  config_list = []
  for corruption in _CORRUPTIONS:
    for severity in range(1, 6):
      config_list.append(
          Cifar10CorruptedConfig(
              name=corruption + '_' + str(severity),
              version=tfds.core.Version(
                  '0.0.1', experiments={tfds.core.Experiment.S3: False}),
              description='Corruption method: ' + corruption +
              ', severity level: ' + str(severity),
              corruption_type=corruption,
              severity=severity,
          ))
  return config_list


class Cifar10Corrupted(tfds.core.GeneratorBasedBuilder):
  """Corrupted Cifar10 dataset."""
  BUILDER_CONFIGS = _make_builder_configs()

  def _info(self):
    """Returns basic information of dataset.

    Returns:
      tfds.core.DatasetInfo.
    """
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=_CIFAR_IMAGE_SIZE),
            'label': tfds.features.ClassLabel(num_classes=_CIFAR_CLASSES),
        }),
        supervised_keys=('image', 'label'),
        urls=['https://github.com/hendrycks/robustness'],
        citation=_CITATION)

  def _split_generators(self, dl_manager):
    """Return the test split of Cifar10.

    Args:
      dl_manager: download manager object.

    Returns:
      test split.
    """
    path = dl_manager.download_and_extract(_DOWNLOAD_URL)
    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=1,
            gen_kwargs={'data_dir': os.path.join(path, _DIRNAME)})
    ]

  def _generate_examples(self, data_dir):
    """Generate corrupted Cifar10 test data.

    Apply corruptions to the raw images according to self.corruption_type.

    Args:
      data_dir: root directory of downloaded dataset

    Yields:
      dictionary with image file and label.
    """
    corruption = self.builder_config.corruption
    severity = self.builder_config.severity

    images_file = os.path.join(data_dir, _CORRUPTIONS_TO_FILENAMES[corruption])
    labels_file = os.path.join(data_dir, _LABELS_FILENAME)

    with tf.io.gfile.GFile(labels_file, mode='rb') as f:
      labels = np.load(f)

    num_images = labels.shape[0] // 5
    # Labels are stacked 5 times so we can just read the first iteration
    labels = labels[:num_images]

    with tf.io.gfile.GFile(images_file, mode='rb') as f:
      images = np.load(f)

    # Slice images corresponding to correct severity level
    images = images[(severity - 1) * num_images:severity * num_images]

    for image, label in zip(images, labels):
      yield {
          'image': image,
          'label': label,
      }
