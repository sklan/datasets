"""MIT Scene Parsing Benchmark (SceneParse150)"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
import tensorflow_datasets as tfds

# BibTeX citation
_CITATION = """
@inproceedings{zhou2017scene,
    title={Scene Parsing through ADE20K Dataset},
    author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
    year={2017}
}
@article{zhou2016semantic,
  title={Semantic understanding of scenes through the ade20k dataset},
  author={Zhou, Bolei and Zhao, Hang and Puig, Xavier and Fidler, Sanja and Barriuso, Adela and Torralba, Antonio},
  journal={arXiv preprint arXiv:1608.05442},
  year={2016}
}
"""


_DESCRIPTION = """
Scene parsing is to segment and parse an image into different image regions associated with semantic categories, such as sky, road, person, and bed. MIT Scene Parsing Benchmark (SceneParse150) provides a standard training and evaluation platform for the algorithms of scene parsing. 
"""

_TRAIN_URL = {
  "images" : "http://placeschallenge.csail.mit.edu/data/ChallengeData2017/images.tar",
  "annotations" : "http://placeschallenge.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar" 
}


_IMAGE_SHAPE = ( 359, 400, 3 )


class SceneParse150(tfds.core.GeneratorBasedBuilder):
  """MIT Scene Parsing Benchmark dataset"""

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
          "image": tfds.features.Image(shape=_IMAGE_SHAPE, encoding_format='jpeg'),
          "annotation": tfds.features.Image(shape=_IMAGE_SHAPE, encoding_format='png')
        }),

        supervised_keys=( "image", "annotation" ),

        urls=["http://sceneparsing.csail.mit.edu/"],

        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    
    dl_paths = dl_manager.download_and_extract({
      "images"      : _TRAIN_URL["images"],
      "annotations" : _TRAIN_URL["annotations"],
    })

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            num_shards=10,
            gen_kwargs={
              "images_dir_path" : os.path.join(dl_paths["images"], "training"),
              "annotations_dir_path": os.path.join(dl_paths["annotations"], "training")
            },
        ),

        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            num_shards=10,
            gen_kwargs={
              "images_dir_path" : os.path.join(dl_paths["images"], "validation"),
              "annotations_dir_path": os.path.join(dl_paths["annotations"], "validation")
            },
        ),
    ]

  def _generate_examples(self, images_dir_path, annotations_dir_path):
    
      
      for image_file in tf.io.gfile.listdir(images_dir_path):
        
        # get the filename
        image_id = os.path.split(image_file)[1].split('.')[0]
        
          
        yield {
          "image" : os.path.join(images_dir_path, "{}.jpg".format(image_id)),
          "annotation" : os.path.join(annotations_dir_path, "{}.png".format(image_id))
        }

