"""AdobeVFR dataset covers 2,383 classes of popular fonts in graphics design."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow_datasets.core import api_utils

_CITATION = """
@article{DBLP:journals/corr/WangYJSABH15a,
  author    = {Zhangyang Wang and
               Jianchao Yang and
               Hailin Jin and
               Eli Shechtman and
               Aseem Agarwala and
               Jonathan Brandt and
               Thomas S. Huang},
  title     = {DeepFont: Identify Your Font from An Image},
  journal   = {CoRR},
  volume    = {abs/1507.03196},
  year      = {2015},
  url       = {http://arxiv.org/abs/1507.03196},
  archivePrefix = {arXiv},
  eprint    = {1507.03196},
  timestamp = {Mon, 13 Aug 2018 16:47:40 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/WangYJSABH15a},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
"""

_DESCRIPTION = """
AdobeVFR dataset covers 2,383 classes of popular fonts in graphics design.
"""


class BCF:
    def __init__(self, filename):
        self._filename = filename
        self._file = tf.io.gfile.GFile(filename, 'rb')
        size = int(np.frombuffer(self._file.read(8))[0])
        file_sizes = np.frombuffer(self._file.read(int(8 * size)))
        self._offsets = np.append(0, np.add.accumulate(file_sizes))

    def get(self, i):
        self._file.seek(int(len(self._offsets) * 8 + self._offsets[i]))
        return self._file.read(int(self._offsets[i + 1] - self._offsets[i]))

    def __del__(self):
        self._file.close()

    def size(self):
        return len(self._offsets) - 1


def read_bcf(path):
    bcf_path = os.path.join(path, 'train.bcf')
    label_path = os.path.join(path, 'train.label')
    labels = read_label(label_path)
    images = BCF(bcf_path)
    return images, labels


def read_label(path):
    with tf.io.gfile.GFile(path, 'rb') as f:
        labels = np.frombuffer(f.read(), dtype=np.uint32)
    return labels


class AdobeVFRConfig(tfds.core.BuilderConfig):
    """BuilderConfig for AdobeVFR."""

    @api_utils.disallow_positional_args
    def __init__(self, mode, **kwargs):
        """BuilderConfig for AdobeVFR Dataset.

        Args:
          mode: raw or bcf
          **kwargs: keyword arguments forwarded to super.
        """
        super(AdobeVFRConfig, self).__init__(**kwargs)
        self.mode = mode


class AdobeVFR(tfds.core.GeneratorBasedBuilder):
    """AdobeVFR dataset covers 2,383 classes of popular fonts in graphics design."""

    VERSION = tfds.core.Version('0.1.0')

    BUILDER_CONFIGS = [
        AdobeVFRConfig(
            name='synthetic',
            description="A collection of images of generated fonts",
            version="0.1.0",
            mode="synthetic",
        ),
        AdobeVFRConfig(
            name='real',
            description="A collection of images of real fonts",
            version="0.1.0",
            mode="real",
        )
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=(None, None, 1)),
                "label": tfds.features.ClassLabel(num_classes=2383)
            }),
            supervised_keys=("image", "label"),
            urls=[
                "https://www.dropbox.com/sh/o320sowg790cxpe/AADDmdwQ08GbciWnaC20oAmna?dl=0"],
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        path = dl_manager.manual_dir
        if self.builder_config.mode == "synthetic":
            path = os.path.join(path, "BCF format")
            if not tf.io.gfile.exists(path):
                message = "You must download the dataset files manually and place the files in: " + dl_manager.manual_dir
                message += "\nLook at datasets/tensorflow_datasets/testing/test_data/fake_examples/adobe_vfr for an example."
                raise AssertionError(message)
            train = os.path.join(path, "VFR_syn_train")
            return [
                tfds.core.SplitGenerator(
                    name=tfds.Split.TRAIN,
                    num_shards=5,
                    gen_kwargs={
                        "path": train
                    },
                ),
            ]

    def _generate_examples(self, path):
        """Yields examples."""
        images, labels = read_bcf(path)
        for i in range(images.size()):
            image = np.asarray(tfds.core.lazy_imports.PIL_Image.open(
                io.BytesIO(images.get(i))))
            image = np.reshape(image,
                               newshape=(image.shape[0], image.shape[1], 1))
            label = labels[i]
            yield {
                "image": image,
                "label": label
            }
