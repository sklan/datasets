"""Tests for Adobe VFR Dataset"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow_datasets import testing
from tensorflow_datasets.image import adobe_vfr


class AdobeVfrTest(testing.DatasetBuilderTestCase):
  DATASET_CLASS = adobe_vfr.AdobeVFR
  BUILDER_CONFIG_NAMES_TO_TEST = ["synthetic"]
  SPLITS = {
      "train": 10, # Number of fake test example
  }

if __name__ == "__main__":
  testing.test_main()

