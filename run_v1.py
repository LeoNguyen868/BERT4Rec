# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
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
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# Suppress TensorFlow INFO and WARNING messages; show only ERRORS
# Also, explicitly tell TensorFlow not to use the GPU if CUDA is problematic
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import modeling
import optimization_v1 as optimization
import tensorflow as tf
import numpy as np
import sys
import pickle
flags = tf.compat.v1.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "train_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "test_input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "checkpointDir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("signature", 'default', "signature_name")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence. "
                     "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

#flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")
flags.DEFINE_integer("batch_size", 32, "Total batch size for training.")

#flags.DEFINE_integer("eval_batch_size", 1, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5,
                   "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 1000, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.compat.v1.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.compat.v1.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_bool("use_pop_random", True, "use pop random negative samples")
flags.DEFINE_string("vocab_filename", None, "vocab filename")
flags.DEFINE_string("user_history_filename", None, "user history filename")


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    # Using Keras layers for new components is preferred, but for now, minimal change:
    # tf.compat.v1.variable_scope can often be managed by Keras layers or python scopes.
    # For this specific structure, direct TF2 equivalents or Keras layers would be a deeper refactor.
    # Assuming this function will be called within a Keras model or custom training step context eventually.

    # Create a dense layer for transformation
    transform_dense_layer = tf.keras.layers.Dense(
        units=bert_config.hidden_size,
        activation=modeling.get_activation(bert_config.hidden_act),
        kernel_initializer=modeling.create_initializer(bert_config.initializer_range),
        name="cls/predictions/transform/dense")
    # Create LayerNorm for transformation
    transform_layer_norm = tf.keras.layers.LayerNormalization(name="cls/predictions/transform/LayerNorm")

    transformed_tensor = transform_dense_layer(input_tensor)
    transformed_tensor = transform_layer_norm(transformed_tensor)

    # Output bias
    # In TF2/Keras, biases are typically part of the Dense layer if use_bias=True.
    # If a separate bias variable is truly needed outside a layer:
    output_bias = tf.Variable(tf.zeros_initializer()(shape=[output_weights.shape[0]]), name="cls/predictions/output_bias")
    
    logits = tf.matmul(transformed_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    one_hot_labels = tf.one_hot(
        label_ids, depth=output_weights.shape[0], dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(input_tensor=log_probs * one_hot_labels, axis=[-1])
    numerator = tf.reduce_sum(input_tensor=label_weights * per_example_loss)
    denominator = tf.reduce_sum(input_tensor=label_weights) + 1e-5
    loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    # This function uses tf.range, tf.reshape, tf.gather which are largely TF2 compatible.
    # Minor adjustment for get_shape_list if it relies on V1 features not in modeling_v2.py
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3) # Assuming modeling.get_shape_list is TF2 compatible
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def _decode_record(record, name_to_features):
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.cast(t, tf.int32)
        example[name] = t
    return example


def create_dataset(input_files, max_seq_length, max_predictions_per_seq, batch_size, 
                     is_training, num_cpu_threads=tf.data.AUTOTUNE):
    name_to_features = {
        "info": tf.io.FixedLenFeature([1], tf.int64),
        "input_ids": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": tf.io.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids": tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights": tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
    }

    if not input_files:
        return None

    d = tf.data.Dataset.from_tensor_slices(input_files)
    if is_training:
        d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))
    
    # TFRecordDataset opens files, so interleave files before map and batch
    d = d.interleave(lambda x: tf.data.TFRecordDataset(x), 
                     cycle_length=num_cpu_threads,
                     num_parallel_calls=tf.data.AUTOTUNE,
                     deterministic=not is_training)

    d = d.map(lambda record: _decode_record(record, name_to_features),
              num_parallel_calls=num_cpu_threads)
    
    if is_training:
        d = d.shuffle(buffer_size=100)
    
    d = d.batch(batch_size, drop_remainder=is_training)
    d = d.prefetch(buffer_size=tf.data.AUTOTUNE)
    return d


class WarmUpAndPolynomialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Applies linear warmup and then polynomial decay."""
    def __init__(self, initial_learning_rate, target_learning_rate, warmup_steps, decay_steps, power, name=None):
        super(WarmUpAndPolynomialDecay, self).__init__()
        self.initial_learning_rate = initial_learning_rate # Typically 0 for warmup start
        self.target_learning_rate = target_learning_rate # This is FLAGS.learning_rate
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps # Total steps for polynomial decay AFTER warmup
        self.power = power
        self.custom_name = name

        # Create the polynomial decay part
        self.polynomial_decay_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=self.target_learning_rate, # Starts after warmup reaches target
            decay_steps=self.decay_steps,
            end_learning_rate=0.0, # Or a configured minimum
            power=self.power,
            cycle=False # Assuming cycle=False from previous setup
        )

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
        
        # Linear warmup phase
        def warmup_lr():
            return self.initial_learning_rate + \
                   (self.target_learning_rate - self.initial_learning_rate) * (step / warmup_steps_float)

        # Polynomial decay phase (adjust step for the decay schedule)
        def polynomial_decay_lr():
            # Step for polynomial decay starts from 0 after warmup is complete
            decay_step = step - warmup_steps_float 
            return self.polynomial_decay_schedule(decay_step)

        learning_rate = tf.cond(
            step < warmup_steps_float,
            warmup_lr,
            polynomial_decay_lr
        )
        return learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "target_learning_rate": self.target_learning_rate,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "power": self.power,
            "name": self.custom_name
        }


def main(_):
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            f"Cannot use sequence length {FLAGS.max_seq_length} because the BERT model "
            f"was only trained up to sequence length {bert_config.max_position_embeddings}")

    if not tf.io.gfile.exists(FLAGS.checkpointDir):
        tf.io.gfile.makedirs(FLAGS.checkpointDir)

    # Create datasets
    train_dataset = None
    if FLAGS.do_train:
        train_input_files = []
        if FLAGS.train_input_file:
            for input_pattern in FLAGS.train_input_file.split(","):
                train_input_files.extend(tf.io.gfile.glob(input_pattern))
        if not train_input_files:
            tf.compat.v1.logging.warning(f"No training input files found for pattern: {FLAGS.train_input_file}")
        else:
            tf.compat.v1.logging.info(f"*** Train Input Files ***")
            for f in train_input_files: tf.compat.v1.logging.info(f"  {f}")
            train_dataset = create_dataset(train_input_files, FLAGS.max_seq_length, 
                                           FLAGS.max_predictions_per_seq, FLAGS.batch_size, is_training=True)
    
    eval_dataset = None
    if FLAGS.do_eval:
        eval_input_files = []
        if FLAGS.test_input_file:
            for input_pattern in FLAGS.test_input_file.split(","):
                eval_input_files.extend(tf.io.gfile.glob(input_pattern))
        if not eval_input_files:
            tf.compat.v1.logging.warning(f"No evaluation input files found for pattern: {FLAGS.test_input_file}")
        else:
            tf.compat.v1.logging.info(f"*** Eval Input Files ***")
            for f in eval_input_files: tf.compat.v1.logging.info(f"  {f}")
            eval_dataset = create_dataset(eval_input_files, FLAGS.max_seq_length, 
                                          FLAGS.max_predictions_per_seq, FLAGS.batch_size, is_training=False)

    # Create Model
    model = modeling.BertModel(
        config=bert_config
        # is_training is handled by the `training` arg in call()
        # use_one_hot_embeddings is no longer a param for Keras BertModel
    )

    # Optimizer
    num_train_steps = FLAGS.num_train_steps if FLAGS.do_train else 0
    num_warmup_steps = FLAGS.num_warmup_steps if FLAGS.do_train else 0
    
    if num_warmup_steps > 0:
        # Total decay steps for PolynomialDecay is after warmup
        polynomial_decay_total_steps = num_train_steps - num_warmup_steps 
        if polynomial_decay_total_steps <= 0:
             # Handle edge case: if all steps are warmup, or more warmup than total steps
             polynomial_decay_total_steps = 1 # Avoid non-positive decay_steps
             tf.compat.v1.logging.warning(
                 f"num_train_steps ({num_train_steps}) is less than or equal to num_warmup_steps ({num_warmup_steps}). \
                 Polynomial decay may not behave as expected."
             )

        learning_rate_schedule = WarmUpAndPolynomialDecay(
            initial_learning_rate=0.0, # Start warmup from 0
            target_learning_rate=FLAGS.learning_rate,
            warmup_steps=num_warmup_steps,
            decay_steps=polynomial_decay_total_steps, 
            power=1.0, # Standard power for linear decay after warmup in many BERT setups
            name="WarmUpAndPolynomialDecay")
    else:
        # No warmup, just polynomial decay
        learning_rate_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
            initial_learning_rate=FLAGS.learning_rate,
            decay_steps=num_train_steps, 
            end_learning_rate=0.0,
            power=1.0,
            cycle=False
        )

    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=learning_rate_schedule, 
        weight_decay=0.01, 
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6
    )

    # Checkpoint Manager
    ckpt = tf.train.Checkpoint(optimizer=optimizer, model=model)
    # To ensure that the model variables are created for checkpointing, build the model
    # by calling it once with dummy data that matches the expected input spec.
    # This is important if restoring a checkpoint before the first training step.
    if FLAGS.max_seq_length > 0: # Only build if max_seq_length is valid
        dummy_input_ids = tf.zeros([1, FLAGS.max_seq_length], dtype=tf.int32)
        dummy_attention_mask = tf.zeros([1, FLAGS.max_seq_length], dtype=tf.int32)
        dummy_token_type_ids = tf.zeros([1, FLAGS.max_seq_length], dtype=tf.int32)
        try:
            model(dummy_input_ids, attention_mask=dummy_attention_mask, token_type_ids=dummy_token_type_ids, training=False)
            tf.compat.v1.logging.info("Model built with dummy input.")
        except Exception as e:
            tf.compat.v1.logging.error(f"Error building model with dummy input: {e}")
            # Potentially raise e or handle as critical error

    ckpt_manager = tf.train.CheckpointManager(ckpt, FLAGS.checkpointDir, max_to_keep=FLAGS.save_checkpoints_steps or 5) # Default to 5 if 0

    if ckpt_manager.latest_checkpoint:
        tf.compat.v1.logging.info(f"Restored from {ckpt_manager.latest_checkpoint}")
        ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial() # Allow partial restore if optimizer state changed etc.
    else:
        tf.compat.v1.logging.info("Initializing from scratch.")

    # Metrics
    train_loss_metric = tf.keras.metrics.Mean("train_loss")
    # For MLM, accuracy might be: masked_lm_accuracy = tf.keras.metrics.Accuracy("masked_lm_accuracy")

    @tf.function
    def train_step(batch_data):
        input_ids = batch_data["input_ids"]
        input_mask = batch_data["input_mask"]
        token_type_ids = batch_data.get("token_type_ids") 
        if token_type_ids is None: 
            pass 
            
        masked_lm_positions = batch_data["masked_lm_positions"]
        masked_lm_ids = batch_data["masked_lm_ids"]
        masked_lm_weights = batch_data["masked_lm_weights"]

        with tf.GradientTape() as tape:
            sequence_output, _ = model(input_ids,
                                       attention_mask=input_mask,
                                       token_type_ids=token_type_ids, 
                                       training=True)
            
            embedding_table = model.get_embedding_table() 
            loss, _, _ = get_masked_lm_output(
                bert_config, sequence_output, embedding_table,
                masked_lm_positions, masked_lm_ids, masked_lm_weights)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        train_loss_metric.update_state(loss)
        return loss

    # Metrics for evaluation
    eval_loss_metric = tf.keras.metrics.Mean("eval_loss")
    # Add other eval metrics if needed, e.g.:
    # eval_masked_lm_accuracy = tf.keras.metrics.Accuracy("eval_masked_lm_accuracy")

    @tf.function
    def eval_step(batch_data):
        input_ids = batch_data["input_ids"]
        input_mask = batch_data["input_mask"]
        token_type_ids = batch_data.get("token_type_ids")
        if token_type_ids is None:
            pass # BertEmbeddings will handle it

        masked_lm_positions = batch_data["masked_lm_positions"]
        masked_lm_ids = batch_data["masked_lm_ids"]
        masked_lm_weights = batch_data["masked_lm_weights"]

        # Forward pass
        sequence_output, _ = model(input_ids,
                                   attention_mask=input_mask,
                                   token_type_ids=token_type_ids,
                                   training=False) # training=False for evaluation
        
        embedding_table = model.get_embedding_table()
        loss, per_example_loss, log_probs = get_masked_lm_output(
            bert_config, sequence_output, embedding_table,
            masked_lm_positions, masked_lm_ids, masked_lm_weights)
        
        eval_loss_metric.update_state(loss)
        
        # Example for accuracy (if masked_lm_predictions are needed)
        # masked_lm_predictions = tf.argmax(input=log_probs, axis=-1, output_type=tf.int32)
        # eval_masked_lm_accuracy.update_state(masked_lm_ids, masked_lm_predictions, sample_weight=masked_lm_weights)
        
        return loss # , per_example_loss, log_probs (if needed for other metrics)

    if FLAGS.do_train:
        if not train_dataset:
            tf.compat.v1.logging.error("Training enabled but no training dataset was created.")
            return
        tf.compat.v1.logging.info("***** Running training *****")
        tf.compat.v1.logging.info(f"  Batch size = {FLAGS.batch_size}")
        tf.compat.v1.logging.info(f"  Num steps = {num_train_steps}")
        
        current_step = optimizer.iterations.numpy()
        train_loss_metric.reset_state() # Reset before new training run or continuation
        for step_idx in range(current_step, num_train_steps):
            # A common way to iterate: make dataset an iterator
            # train_iter = iter(train_dataset) 
            # batch = next(train_iter)
            # Or if dataset is already repeating and we just take N steps:
            try:
                batch = next(iter(train_dataset)) 
            except StopIteration:
                # This shouldn't happen if train_dataset.repeat() is used and num_train_steps is finite.
                # If it does, re-initialize iterator (though this implies epochs, not just global steps)
                train_iter = iter(train_dataset) # Re-init iterator if dataset is not infinite
                batch = next(train_iter)
            
            loss_val = train_step(batch)
            
            if step_idx % 100 == 0: 
                tf.compat.v1.logging.info(f"Step {optimizer.iterations.numpy()}/{num_train_steps}, Loss: {train_loss_metric.result()}")
            
            if step_idx > 0 and step_idx % FLAGS.save_checkpoints_steps == 0:
                saved_path = ckpt_manager.save()
                tf.compat.v1.logging.info(f"Saved checkpoint for step {optimizer.iterations.numpy()}: {saved_path}")
        
        final_train_loss = train_loss_metric.result()
        tf.compat.v1.logging.info(f"Training finished. Final Loss: {final_train_loss}")
        # train_loss_metric.reset_states() # Done at the start of training

    if FLAGS.do_eval:
        if not eval_dataset:
            tf.compat.v1.logging.error("Evaluation enabled but no eval dataset was created.")
            return
        tf.compat.v1.logging.info("***** Running evaluation *****")
        tf.compat.v1.logging.info(f"  Batch size = {FLAGS.batch_size}")
        # max_eval_steps is not directly used in dataset iteration but can limit the eval
        # For a full evaluation, iterate over the entire eval_dataset once.
        
        eval_loss_metric.reset_state()
        # eval_masked_lm_accuracy.reset_states() # if using accuracy

        eval_steps_count = 0
        for batch_idx, batch in enumerate(eval_dataset):
            if FLAGS.max_eval_steps and batch_idx >= FLAGS.max_eval_steps:
                break
            eval_step(batch)
            eval_steps_count += 1
            if batch_idx % 100 == 0:
                 tf.compat.v1.logging.info(f"Eval Step {batch_idx}, Current Eval Loss: {eval_loss_metric.result()}")

        tf.compat.v1.logging.info("***** Eval results *****")
        tf.compat.v1.logging.info(f"Evaluation on {eval_steps_count} batches complete.")
        tf.compat.v1.logging.info(f"  Eval Loss = {eval_loss_metric.result()}")
        # if eval_masked_lm_accuracy:
        #    tf.compat.v1.logging.info(f"  Eval Masked LM Accuracy = {eval_masked_lm_accuracy.result()}")
        eval_loss_metric.reset_state()
        # eval_masked_lm_accuracy.reset_states()

    tf.compat.v1.logging.info("Script execution finished.") # Updated log message


if __name__ == "__main__":
    #flags.mark_flag_as_required("bert_config_file")
    #flags.mark_flag_as_required("checkpointDir")
    tf.compat.v1.app.run() 