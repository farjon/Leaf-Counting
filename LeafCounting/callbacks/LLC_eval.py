"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from ..utils.eval_LCC import evaluate

class Evaluate_LLCtype(keras.callbacks.Callback):
    def __init__(self, generator, save_path=None, tensorboard=None, verbose=1):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator       : The generator that represents the dataset to evaluate.
            iou_threshold   : The threshold used to consider when a detection is positive or negative.
            score_threshold : The score confidence threshold to use for detections.
            max_detections  : The maximum number of detections to use per image.
            save_path       : The path to save images with visualized detections to.
            tensorboard     : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            verbose         : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.verbose         = verbose

        super(Evaluate_LLCtype, self).__init__()

    def on_epoch_begin(self, epoch, logs=None):
        self.generator.set_epoch(epoch)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # start evaluation
        self.CountDiff, self.AbsCountDiff, self.CountAgreement, self.mse = evaluate(
            self.generator.get_option(),
            self.generator.get_csv_leaf_number_file(),
            self.generator,
            self.model,
            save_path=self.save_path
        )


        if self.tensorboard is not None and self.tensorboard.writer is not None:
            import tensorflow as tf
            summary = tf.Summary()

            summary_value = summary.value.add()
            summary_value.simple_value = self.CountDiff
            summary_value.tag = "CountDiff"

            summary_value = summary.value.add()
            summary_value.simple_value = self.AbsCountDiff
            summary_value.tag = "AbsCountDiff"

            summary_value = summary.value.add()
            summary_value.simple_value = self.CountAgreement
            summary_value.tag = "CountAgreement"

            summary_value = summary.value.add()
            summary_value.simple_value = self.mse
            summary_value.tag = "mse"

            self.tensorboard.writer.add_summary(summary, epoch)

        logs['CountDiff'] = self.CountDiff
        logs['AbsCountDiff'] = self.AbsCountDiff
        logs['CountAgreement'] = self.CountAgreement
        logs['mse'] = self.mse

        if self.verbose == 1:
            print("CountDiff:", self.CountDiff, "AbsCountDiff", self.AbsCountDiff, "CountAgreement", self.CountAgreement, "mse", self.mse)
