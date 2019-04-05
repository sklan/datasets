# Generating big datasets with Apache Beam

Some datasets are too big to be processed on a single machine. `tfds` support
[Apache Beam](https://beam.apache.org/) which allow to generate the data in a
distributed environement.

Note: This mode is still experimental, so the API may change in the future
depending on user feedback. Do not hesitate to
[submit your feedback](https://github.com/tensorflow/datasets/issues/new?assignees=&labels=enhancement&template=feature_request.md&title=).

Warning: Due to some Apache Beam bug, generation won't work in Python3 for
version 2.11 or previous. A
[fix](https://github.com/apache/beam/commit/46a70a8b4691a169c9e018299765e1bdb88239f4)
will be released in the next Apache Beam version.

*   [Instruction](#instruction)
    *   [Prerequisite](#prerequisite)
    *   [Changes to make](#changes-to-make)
    *   [Example](#example)
    *   [Run your pipeline](#run-your-pipeline)

## Instruction

### Prerequisite

In order to write Apache Beam datasets, you should be familiar with the
following concepts:

*   Be familiar with the
    [tfds dataset creation guide](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md)
    as most of the content still apply for Beam dataset
*   Get an instroduction on Apache Beam, with the
    [Beam programming guide](https://beam.apache.org/documentation/programming-guide/)
*   If you want to generate your dataset using Cloud Dataflow, read the
    [Google Cloud Documentation](https://cloud.google.com/dataflow/docs/quickstarts/quickstart-python)
    and the
    [Apache Beam dependency guide](https://beam.apache.org/documentation/sdks/python-pipeline-dependencies/)

### Changes to make

If you are familiar with the
[dataset creation guide](https://github.com/tensorflow/datasets/tree/master/docs/add_dataset.md),
adding a new Beam base dataset only requires a few modifications:

*   Dataset inherit from `tfds.core.BeamBasedBuilder` instead of
    `tfds.core.GeneratorBasedBuilder`
*   Beam datasets should implement the abstract method `def
    _build_pcollection(self, **kwargs)` instead of the `def
    _generate_examples(self, **kwargs)`. This method should return the
    `beam.PCollection` associated with the split.
*   Writing unit-test for Beam dataset is identical to other datasets.

Some additional conciderations:

*   Use `tfds.core.lazy_imports` to import Apache Beam. By using lazy
    dependency, users can use the dataset without installing Beam after the
    dataset has been generated once.
*   Be careful with python closure. When running the pipeline, the `beam.Map`
    and `beam.DoFn` functions are serialized using pickle and sent to all
    workers. This can create bugs, for instance if you are using mutable object
    in your functions which have been declared outside of those. If you are
    getting `pickle` errors, this is probably related to Python closure.

### Example

Here is an example of a Beam dataset. For a more complicated real example, have
a look at the
[wikipedia dataset](https://github.com/tensorflow/datasets/tree/master/tensorflow_datasets/text/wikipedia.py).

```python
class DummyBeamDataset(tfds.core.BeamBasedBuilder):

  VERSION = tfds.core.Version('1.0.0')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(shape=(16, 16, 1)),
            'label': tfds.features.ClassLabel(names=['dog', 'cat']),
        }),
    )

  def _split_generators(self, dl_manager):
    ...
    return [
        tfds.core.SplitGenerator(
            name=tfds..Split.TRAIN,
            num_shards=100,
            gen_kwargs=dict(file_dir='path/to/train_data/'),
        ),
        splits_lib.SplitGenerator(
            name=tfds..Split.TEST,
            num_shards=10,
            gen_kwargs=dict(file_dir='path/to/test_data/'),
        ),
    ]

  def _build_pcollection(self, pipeline, file_dir):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(filename):
      return {
          'image': os.path.join(file_dir, filename),
          'label': filename.split('.')[1],  # Extract label: "0010102.dog.jpeg"
      }

    return (
        pipeline
        | beam.Create(tf.io.gfile.listdir(file_dir))
        | beam.Map(_process_example)
    )

```

### Run your pipeline

To generate the dataset on Beam, the API is the same as for other datasets, but
you have to pass the Beam options or runner to the `DownloadConfig`.

```
dl_config = tfds.download.DownloadConfig(
    beam_options=beam.options.pipeline_options.PipelineOptions()
    beam_runner=None,
)

builder = tfds.builder('wikipedia')
builder.download_and_prepare(
    download_dir=FLAGS.download_dir,
    download_config=dl_config,
)
```

To run your script locally using the default Apache Beam runner, the command is
the same as per other datasets:

```
python -m tensorflow_datasets.scripts.download_and_prepare \
  --register_checksums \
  --datasets=my_new_dataset
```

TODO: Add instructions to run with `Cloud Dataflow`
