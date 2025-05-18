# -*- coding: UTF-8 -*-
import os
import codecs

import collections
import random

import sys

import tensorflow as tf

import six

from util import *
from vocab import *
import pickle
import multiprocessing
import time


random_seed = 12345
short_seq_prob = 0  # Probability of creating sequences which are shorter than the maximum length。

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("signature", 'default', "signature_name")

flags.DEFINE_integer(
    "pool_size", 10,
    "multiprocesses pool size.")

flags.DEFINE_integer(
    "max_seq_length", 200,
    "max sequence length.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "max_predictions_per_seq.")

flags.DEFINE_float(
    "masked_lm_prob", 0.15,
    "Masked LM probability.")

flags.DEFINE_float(
    "mask_prob", 1.0,
    "mask probabaility")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("prop_sliding_window", 0.1, "sliding window step size.")
    
flags.DEFINE_string(
    "data_dir", './data/',
    "data dir.")

flags.DEFINE_string(
    "dataset_name", 'ml-1m',
    "dataset name.")


def printable_text(text):
    """Returns text encoded in a way suitable for print or `tf.logging`."""

    # These functions want `str` for both Python2 and Python3, but in one case
    # it's a Unicode string and in the other it's a byte string.
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text
        elif isinstance(text, unicode):
            return text.encode("utf-8")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, info, tokens, masked_lm_positions, masked_lm_labels):
        self.info = info  # info = [user]
        self.tokens = tokens
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "info: %s\n" % (" ".join([printable_text(x) for x in self.info]))
        s += "tokens: %s\n" % (
            " ".join([printable_text(x) for x in self.tokens]))
        s += "masked_lm_positions: %s\n" % (
            " ".join([str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (
            " ".join([printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files):
    """Create TF example files from `TrainingInstance`s."""
    writers = []
    for output_file in output_files:
        writers.append(tf.compat.v1.python_io.TFRecordWriter(output_file))

    writer_index = 0

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        try:
            input_ids = vocab.convert_tokens_to_ids(instance.tokens)
        except:
            print(instance)

        input_mask = [1] * len(input_ids)
        assert len(input_ids) <= max_seq_length

        input_ids += [0] * (max_seq_length - len(input_ids))
        input_mask += [0] * (max_seq_length - len(input_mask))

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = vocab.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        masked_lm_positions += [0] * (max_predictions_per_seq - len(masked_lm_positions))
        masked_lm_ids += [0] * (max_predictions_per_seq - len(masked_lm_ids))
        masked_lm_weights += [0.0] * (max_predictions_per_seq - len(masked_lm_weights))

        features = collections.OrderedDict()
        features["info"] = create_int_feature(instance.info)
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["masked_lm_positions"] = create_int_feature(
            masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)

        tf_example = tf.train.Example(
            features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.compat.v1.logging.info("*** Example ***")
            tf.compat.v1.logging.info("tokens: %s" % " ".join(
                [printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.compat.v1.logging.info("%s: %s" % (feature_name,
                                            " ".join([str(x)
                                                      for x in values])))

    for writer in writers:
        writer.close()

    tf.compat.v1.logging.info("Wrote %d total instances", total_written)


def create_int_feature(values):
    feature = tf.train.Feature(
        int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(
        float_list=tf.train.FloatList(value=list(values)))
    return feature


def create_training_instances(all_documents_raw,
                              max_seq_length,
                              dupe_factor,
                              short_seq_prob,
                              masked_lm_prob,
                              max_predictions_per_seq,
                              rng,
                              vocab,
                              mask_prob,
                              prop_sliding_window,
                              pool_size,
                              force_last=False):
    """Create `TrainingInstance`s from raw text."""
    all_documents = {}
    current_dupe_factor = dupe_factor 
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting instance creation process... force_last={force_last}, dupe_factor for this call={current_dupe_factor}")

    if force_last:
        max_num_tokens = max_seq_length
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue
            all_documents[user] = [item_seq[-max_num_tokens:]]
    else:
        max_num_tokens = max_seq_length  # we need two sentence

        sliding_step = (int)(
            prop_sliding_window *
            max_num_tokens) if prop_sliding_window != -1.0 else max_num_tokens
        for user, item_seq in all_documents_raw.items():
            if len(item_seq) == 0:
                print("got empty seq:" + user)
                continue

            #todo: add slide
            if len(item_seq) <= max_num_tokens:
                all_documents[user] = [item_seq]
            else:
                beg_idx = range(len(item_seq) - max_num_tokens, 0, -sliding_step)
                #beg_idx.append(0)
                all_documents[user] = [item_seq[i:i + max_num_tokens] for i in beg_idx[::-1]]
                #fixed bug for sequence order
                #all_documents[user].append(item_seq[:max_num_tokens])

    instances = []
    if force_last:
        for user in all_documents:
            instances.extend(
                mask_last(all_documents, user, max_seq_length,
                                short_seq_prob, masked_lm_prob,
                                max_predictions_per_seq, vocab, rng))
        print("num of instance:" + str(len(instances)))
        return instances

    # multithread
    start_time = time.time()
    # Temporarily reduce pool_size for debugging, can be passed from main call later
    # pool_size = 1 # DEBUG: Reduced pool size
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Initializing multiprocessing pool with size: {pool_size}")
    pool = multiprocessing.Pool(processes=pool_size)
    #rng.shuffle(all_documents.keys()) #this is shuffle user
    user_list = list(all_documents.keys())
    #print("user_list", user_list)
    #random.shuffle(user_list)
    results = []

    def log_result(result):
        results.extend(result)

    for step in range(current_dupe_factor):
        random.shuffle(user_list)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting dupe_factor step {step+1}/{current_dupe_factor} for {len(user_list)} users.")
        for idx, user in enumerate(user_list):
            if idx % 100 == 0: # Log every 100 users submitted to pool
                print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Submitting user {idx+1}/{len(user_list)} (User ID: {user}) to pool in step {step+1}.")
            #print("user", user)
            pool.apply_async(
                create_instances_threading,
                args=(all_documents, user, max_seq_length, short_seq_prob,
                      masked_lm_prob, max_predictions_per_seq, vocab, rng,
                      mask_prob, step),
                callback=log_result)

    pool.close()
    pool.join()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Multiprocessing pool finished.")

    # for result in results:
    #     instances.extend(result)
    instances = results

    print("MULTIPROCESSING CREATE INSTANCES", time.time() - start_time)
    rng.shuffle(instances)
    return instances


def create_instances_threading(all_documents, user, max_seq_length, short_seq_prob,
                               masked_lm_prob, max_predictions_per_seq, vocab, rng,
                               mask_prob, step):
    # Verbose logging for debugging hangs
    print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] THREADING: User {user}, Step {step+1} - START")
    document_instances = []
    max_num_tokens = max_seq_length - 1

    for doc_index in range(len(all_documents[user])):
        # print(f"      [{time.strftime('%Y-%m-%d %H:%M:%S')}] THREADING: User {user}, Step {step+1}, Doc {doc_index} - Processing document")
        document_instances.extend(
            create_instances_from_document_train(
                all_documents, user, doc_index, max_num_tokens,
                short_seq_prob, masked_lm_prob, max_predictions_per_seq,
                vocab, rng, mask_prob, step)) # Pass step for logging
    print(f"    [{time.strftime('%Y-%m-%d %H:%M:%S')}] THREADING: User {user}, Step {step+1} - END, instances created: {len(document_instances)}")
    return document_instances


def mask_last(
        all_documents, user, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user][0]  # Use the last one

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length -1

    # We *sometimes* want to use shorter sequences to minimize the mismatch
    # between pre-training and fine-tuning. However, we select the documents
    # randomly based on the input length to make sure that the distribution
    # of sentence lengths doesn't change.
    # target_seq_length = max_num_tokens
    # if rng.random() < short_seq_prob:
    # target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    # print("document", document)
    # current_chunk = []
    # current_length = 0
    # i = 0
    # while i < len(document):
    #     segment = document[i]
    #     current_chunk.append(segment)
    #     current_length += len(segment)
    #     if i == len(document) - 1 or current_length >= target_seq_length:
    #         if current_chunk:
    #             # `a_end` is how many segments from `current_chunk` go into the `A`
    #             # sentence.
    #             a_end = 1
    #             if len(current_chunk) >= 2:
    #                 a_end = rng.randint(1, len(current_chunk) - 1)
    #
    #             tokens_a = []
    #             for j in range(a_end):
    #                 tokens_a.extend(current_chunk[j])
    #
    #             tokens_b = []

    info = [int(user.split('_')[1])]
    tokens = list(document)
    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
    instances.append(instance)

    #             current_chunk = []
    #     i += 1
    return instances


def create_instances_from_document_test(all_documents, user, max_seq_length):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[user][0]  # Use the last one

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 1

    # We *sometimes* want to use shorter sequences to minimize the mismatch
    # between pre-training and fine-tuning. However, we select the documents
    # randomly based on the input length to make sure that the distribution
    # of sentence lengths doesn't change.
    # target_seq_length = max_num_tokens
    # if rng.random() < short_seq_prob:
    # target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []

    info = [int(user.split('_')[1])]
    #tokens_a = document
    #tokens_b = []
    tokens = list(document)
    # print(user, tokens_a)
    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions_force_last(tokens)
    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
    instances.append(instance)
    # print(instance)

    return instances


def create_instances_from_document_train(
        all_documents, user, doc_index, max_seq_length, short_seq_prob, masked_lm_prob,
        max_predictions_per_seq, vocab, rng, mask_prob, step_for_log):
    print(f"      [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - START")
    document = all_documents[user][doc_index]
    max_num_tokens = max_seq_length -1

    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    info = [int(user.split('_')[1])]
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - Preparing tokens. Original document length: {len(document)}")
    tokens = list(document) 
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - Tokens list created. Length: {len(tokens)}. Vocab keys to list next.")
    vocab_keys_list = list(vocab.keys())
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - Vocab keys list created. Length: {len(vocab_keys_list)}. Calling create_masked_lm_predictions.")

    (tokens, masked_lm_positions,
     masked_lm_labels) = create_masked_lm_predictions(
         tokens, masked_lm_prob, max_predictions_per_seq, vocab_keys_list, rng,
         mask_prob, user, step_for_log, doc_index)
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - Returned from create_masked_lm_predictions.")

    instance = TrainingInstance(
        info=info,
        tokens=tokens,
        masked_lm_positions=masked_lm_positions,
        masked_lm_labels=masked_lm_labels)
    print(f"      [{time.strftime('%Y-%m-%d %H:%M:%S')}] DOC_TRAIN: User {user}, Step {step_for_log+1}, Doc {doc_index} - END, instances: 1")
    return [instance]


def create_masked_lm_predictions_force_last(tokens):
    """Creates the predictions for the masked LM objective for forcing last item."""

    last_index = -1
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Comment here to not mask the last element
        # if (i == len(tokens) - 1):
        #     last_index = i
    # print("last_index", last_index)
    # print("tokens", tokens)
    output_tokens = list(tokens)
    # output_tokens[last_index] = "[MASK]"
    # if tokens[-1] != '[SEP]':
    #   tokens.append('[SEP]')
    if output_tokens[-1] == '[SEP]':
        last_index = len(output_tokens) - 2
        output_tokens[-2] = "[MASK]"
    else: # this case should not happen
        last_index = len(output_tokens) - 1
        output_tokens[-1] = "[MASK]"
    # else:
    #     last_index = len(output_tokens) - 1
    #     output_tokens.append("[MASK]")
    #     tokens.append("[SEP]")


    masked_lm_positions = [last_index]
    masked_lm_labels = [tokens[last_index]]  #
    # print("masked_lm_labels", masked_lm_labels)

    return (output_tokens, masked_lm_positions, masked_lm_labels)


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng,
                                 mask_prob,
                                 # For logging purposes
                                 user_for_log, step_for_log, doc_index_for_log):
    """Creates the predictions for the masked LM objective."""
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] MASK_PRED: User {user_for_log}, Step {step_for_log+1}, Doc {doc_index_for_log} - START. Tokens length: {len(tokens)}")

    cand_indexes = []
    # [CLS] and [SEP] will be excluded
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_pos, index in enumerate(cand_indexes):
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        # print(f"          [{time.strftime('%Y-%m-%d %H:%M:%S')}] MASK_PRED: User {user_for_log}, Step {step_for_log+1}, Doc {doc_index_for_log} - Masking loop {index_pos}, Index {index}")

        masked_token = None
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)
    # if len(masked_lm_positions) == 0: 
    #     # backoff to the last one
    #     index = len(tokens) -1
    #     output_tokens[index] = "[MASK]"
    #     masked_lm_positions.append(index)
    #     masked_lm_labels.append(tokens[index])
    #     print("Cannot find a token to mask")
    print(f"        [{time.strftime('%Y-%m-%d %H:%M:%S')}] MASK_PRED: User {user_for_log}, Step {step_for_log+1}, Doc {doc_index_for_log} - END. Predicted: {len(masked_lms)}")
    return (output_tokens, masked_lm_positions, masked_lm_labels)






def gen_samples(data,
                output_filename,
                rng,
                vocab,
                max_seq_length,
                dupe_factor,
                short_seq_prob,
                mask_prob,
                masked_lm_prob,
                max_predictions_per_seq,
                prop_sliding_window,
                pool_size,
                force_last=False):
    # create train
    instances = create_training_instances(
        data, max_seq_length, dupe_factor, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, rng, vocab, mask_prob,
        prop_sliding_window, pool_size, force_last)

    tf.compat.v1.logging.info("number of instances: %s", len(instances))

    output_files = [output_filename]
    tf.compat.v1.logging.info("*** Writing to output files ***")
    for output_file in output_files:
        tf.compat.v1.logging.info("  %s", output_file)

    write_instance_to_example_files(instances, max_seq_length,
                                    max_predictions_per_seq, vocab,
                                    output_files)


def main():
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

    #     flags.DEFINE_string("input_file", './data/ml-1m.txt', "input file")
    #     flags.DEFINE_string("output_file", './data/ml-1m.tfrecord', "output file")
    #     flags.DEFINE_string("vocab_file_name", './data/ml-1m_vocab.pkl', "vocab file name")
    #     flags.DEFINE_string("user_history_file_name", './data/ml-1m_history.pkl', "user history file name")

    #     flags.DEFINE_bool(
    #         "do_lower_case", True,
    #         "Whether to lower case the input text. Should be True for uncased "
    #         "models and False for cased models.")
    #     flags.DEFINE_integer("max_seq_length", 200, "Maximum A sequence length.")
    #     flags.DEFINE_integer("max_predictions_per_seq", 20,
    #                          "Maximum number of masked LM predictions per sequence.")
    #     flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
    #     flags.DEFINE_integer(
    #         "dupe_factor", 10,
    #         "Number of times to duplicate the input data (with different masks).")
    #     flags.DEFINE_float("masked_lm_prob", 0.2, "Masked LM probability.")
    #     flags.DEFINE_float(
    #         "short_seq_prob", 0.1,
    #         "Probability of creating sequences which are shorter than the "
    #         "maximum length.")

    if FLAGS.max_predictions_per_seq is None:
        raise ValueError("max_predictions_per_seq is None")

    if FLAGS.prop_sliding_window is None:
        FLAGS.prop_sliding_window = (FLAGS.max_seq_length *
                                 FLAGS.masked_lm_prob) / FLAGS.max_seq_length

    print("prop_sliding_window", FLAGS.prop_sliding_window)

    # data_dir = './data/'
    # dataset_name = 'ml-1m'

    if FLAGS.dataset_name == 'ml-1m':
        data_file = FLAGS.data_dir + 'ml-1m.txt'
    elif FLAGS.dataset_name == 'ml-20m':
        data_file = FLAGS.data_dir + 'ml-20m.txt'
    elif FLAGS.dataset_name == 'beauty':
        data_file = FLAGS.data_dir + 'beauty.txt'
    elif FLAGS.dataset_name == 'steam':
        data_file = FLAGS.data_dir + 'steam.txt'
    else:
        print("error dataset_name, uses beauty by default")
        data_file = FLAGS.data_dir + 'beauty.txt'

    output_filename = FLAGS.data_dir + FLAGS.dataset_name + "_" + FLAGS.signature + ".train.tfrecord"
    vocab_file = FLAGS.data_dir + FLAGS.dataset_name + "_" + FLAGS.signature + ".vocab"
    user_history_file = FLAGS.data_dir + FLAGS.dataset_name + "_" + FLAGS.signature + ".history"

    print("output_filename", output_filename)
    print("vocab_file", vocab_file)
    print("user_history_file", user_history_file)

    rng = random.Random(random_seed)
    data = collections.defaultdict(list)
    user_history = collections.defaultdict(list)

    # Always load data from data_file
    print("load data from", data_file)
    f = open(data_file, 'r')
    start_load_time = time.time()
    for line_idx, line in enumerate(f):
        if (line_idx + 1) % 200000 == 0: # Log every 200,000 lines
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Processing line {line_idx+1} from {data_file}")
        parts = line.rstrip().split(' ') 
        if len(parts) == 2: 
            user, item = parts[0], parts[1]
            data["user_"+user].append("item_"+item)
            # Populate user_history here as well if it's meant to be the raw sequences
            # Or, it might be that user_history is only for already processed/pickled history
            user_history["user_"+user].append("item_"+item) 
        elif len(parts) == 4:
            user, item, rating, timestamp = parts
            data["user_"+user].append("item_"+item)
            user_history["user_"+user].append("item_"+item)
        else:
            tf.compat.v1.logging.warning(f"Skipping malformed line: {line.rstrip()}")
            continue
    f.close()
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Finished loading {data_file} in {time.time() - start_load_time:.2f} seconds. {len(data)} users loaded.")

    if os.path.isfile(user_history_file) and os.path.isfile(vocab_file):
        print("load user history from", user_history_file)
        # user_history is already populated from data_file, 
        # so loading from pickle might be redundant or for a different purpose.
        # For now, let's assume the user_history from data_file is the source of truth
        # and pickling is for caching. If this assumption is wrong, this part may need review.
        # with open(user_history_file, 'rb') as pkl_file:
        #     user_history = pickle.load(pkl_file) # Potentially overwrite or merge
        print("load vocab from", vocab_file)
        with open(vocab_file, 'rb') as pkl_file:
            vocab = pickle.load(pkl_file)
    else:
        vocab = FreqVocab(data)
        with open(vocab_file, 'wb') as output_file:
            pickle.dump(vocab, output_file, protocol=2)
        # user_history is already populated, pickle it if creating for the first time
        with open(user_history_file, 'wb') as output_file:
            pickle.dump(user_history, output_file, protocol=2)

    # for (user_id, item_seq) in data.items():
    #     print(user_id, item_seq)
    # print("vocab", vocab.counter)

    # for train
    gen_samples(
        data,
        output_filename,
        rng,
        vocab,
        max_seq_length=FLAGS.max_seq_length,
        dupe_factor=FLAGS.dupe_factor,
        short_seq_prob=short_seq_prob,
        mask_prob=FLAGS.mask_prob,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        prop_sliding_window=FLAGS.prop_sliding_window,
        pool_size=FLAGS.pool_size)

    print("train finish")

    # for test
    output_filename = FLAGS.data_dir + FLAGS.dataset_name + "_" + FLAGS.signature + ".test.tfrecord"
    gen_samples(
        data,
        output_filename,
        rng,
        vocab,
        max_seq_length=FLAGS.max_seq_length,
        dupe_factor=1,
        short_seq_prob=short_seq_prob,
        mask_prob=FLAGS.mask_prob,
        masked_lm_prob=FLAGS.masked_lm_prob,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        prop_sliding_window=FLAGS.prop_sliding_window,
        pool_size=FLAGS.pool_size,
        force_last=True)

    print("test finish")


if __name__ == "__main__":
    #flags.DEFINE_string("input_file", './data/ml-1m.txt', "input file")
    #flags.DEFINE_string("output_file", './data/ml-1m.tfrecord', "output file")
    #flags.DEFINE_string("vocab_file_name", './data/ml-1m_vocab.pkl', "vocab file name")
    #flags.DEFINE_string("user_history_file_name", './data/ml-1m_history.pkl', "user history file name")

    #flags.DEFINE_bool(
    #    "do_lower_case", True,
    #    "Whether to lower case the input text. Should be True for uncased "
    #    "models and False for cased models.")
    #flags.DEFINE_integer("max_seq_length", 200, "Maximum A sequence length.")
    #flags.DEFINE_integer("max_predictions_per_seq", 20,
    #                     "Maximum number of masked LM predictions per sequence.")
    #flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")
    #flags.DEFINE_integer(
    #    "dupe_factor", 10,
    #    "Number of times to duplicate the input data (with different masks).")
    #flags.DEFINE_float("masked_lm_prob", 0.2, "Masked LM probability.")
    #flags.DEFINE_float(
    #    "short_seq_prob", 0.1,
    #    "Probability of creating sequences which are shorter than the "
    #    "maximum length.")
    main() 