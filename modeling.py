# coding=utf-8

"""The main BERT model and related functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import six
import tensorflow as tf


class BertConfig(object):
    """Configuration for `BertModel`."""

    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=16,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
        vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
        hidden_size: Size of the encoder layers and the pooler layer.
        num_hidden_layers: Number of hidden layers in the Transformer encoder.
        num_attention_heads: Number of attention heads for each attention layer in
            the Transformer encoder.
        intermediate_size: The size of the "intermediate" (i.e., feed-forward)
            layer in the Transformer encoder.
        hidden_act: The non-linear activation function (function or string) in the
            encoder and pooler.
        hidden_dropout_prob: The dropout probability for all fully connected
            layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob: The dropout ratio for the attention
            probabilities.
        max_position_embeddings: The maximum sequence length that this model might
            ever be used with. Typically set this to something large just in case
            (e.g., 512 or 1024 or 2048).
        type_vocab_size: The vocabulary size of the `token_type_ids` passed into
            `BertModel`.
        initializer_range: The stdev of the truncated_normal_initializer for
            initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with tf.io.gfile.GFile(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertEmbeddings(tf.keras.layers.Layer):
    """BERT Embedding layer that combines word, position, and token type embeddings."""
    def __init__(self, config, name="embeddings", **kwargs):
        super(BertEmbeddings, self).__init__(name=name, **kwargs)
        self.config = config
        self.word_embeddings = tf.keras.layers.Embedding(
            config.vocab_size,
            config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="word_embeddings")
        self.position_embeddings = tf.keras.layers.Embedding(
            config.max_position_embeddings,
            config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="position_embeddings")
        self.token_type_embeddings = tf.keras.layers.Embedding(
            config.type_vocab_size,
            config.hidden_size,
            embeddings_initializer=create_initializer(config.initializer_range),
            name="token_type_embeddings")
        
        self.layer_norm = tf.keras.layers.LayerNormalization(
            epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, input_ids, token_type_ids=None, training=None):
        input_shape = get_shape_list(input_ids)
        seq_length = input_shape[1]

        position_ids = tf.range(seq_length, dtype=tf.int32)[tf.newaxis, :]
        if token_type_ids is None:
            token_type_ids = tf.zeros_like(input_ids, dtype=tf.int32)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings_val = self.position_embeddings(position_ids)
        token_type_embeddings_val = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings_val + token_type_embeddings_val
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings, training=training)
        return embeddings

    def get_embedding_table(self):
        return self.word_embeddings.embeddings


class BertSelfAttention(tf.keras.layers.Layer):
    def __init__(self, config, name="self", **kwargs):
        super(BertSelfAttention, self).__init__(name=name, **kwargs)
        self.num_attention_heads = config.num_attention_heads
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of "
                f"attention heads ({config.num_attention_heads})"
            )
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_layer = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=create_initializer(config.initializer_range), name="query"
        )
        self.key_layer = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=create_initializer(config.initializer_range), name="key"
        )
        self.value_layer = tf.keras.layers.Dense(
            self.all_head_size, kernel_initializer=create_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, input_tensor, batch_size, seq_length):
        output_tensor = tf.reshape(
            input_tensor, (batch_size, seq_length, self.num_attention_heads, self.attention_head_size)
        )
        return tf.transpose(output_tensor, perm=[0, 2, 1, 3])

    def call(self, hidden_states, attention_mask, training=None):
        # hidden_states: [batch_size, seq_length, hidden_size]
        # attention_mask: [batch_size, 1, 1, seq_length] (additive mask)
        input_shape = get_shape_list(hidden_states)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        mixed_query_layer = self.query_layer(hidden_states)
        mixed_key_layer = self.key_layer(hidden_states)
        mixed_value_layer = self.value_layer(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size, seq_length)
        key_layer = self.transpose_for_scores(mixed_key_layer, batch_size, seq_length)
        value_layer = self.transpose_for_scores(mixed_value_layer, batch_size, seq_length)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
        attention_scores = attention_scores / math.sqrt(float(self.attention_head_size))

        if attention_mask is not None:
            # The attention_mask is [B, F, T] from create_attention_mask_from_input_mask
            # It needs to be [B, 1, F, T] for broadcasting with attention_scores [B, N, F, T]
            attention_mask_expanded = tf.expand_dims(attention_mask, axis=[1])
            attention_scores = attention_scores + attention_mask_expanded

        attention_probs = tf.nn.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs, training=training)

        context_layer = tf.matmul(attention_probs, value_layer)
        context_layer = tf.transpose(context_layer, perm=[0, 2, 1, 3])
        context_layer = tf.reshape(
            context_layer, (batch_size, seq_length, self.all_head_size)
        )
        return context_layer


class BertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config, name="output", **kwargs):
        super(BertSelfOutput, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, kernel_initializer=create_initializer(config.initializer_range), name="dense"
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, hidden_states, input_tensor, training=None):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states, training=training)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(tf.keras.layers.Layer):
    def __init__(self, config, name="attention", **kwargs):
        super(BertAttention, self).__init__(name=name, **kwargs)
        self.self_attention = BertSelfAttention(config, name="self")
        self.output_layer = BertSelfOutput(config, name="output") 

    def call(self, hidden_states, attention_mask, training=None):
        self_attention_outputs = self.self_attention(hidden_states, attention_mask, training=training)
        attention_output = self.output_layer(self_attention_outputs, hidden_states, training=training)
        return attention_output


class BertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config, name="intermediate", **kwargs):
        super(BertIntermediate, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(
            config.intermediate_size, 
            kernel_initializer=create_initializer(config.initializer_range),
            activation=get_activation(config.hidden_act), 
            name="dense" # Original scope: bert/encoder/layer_X/intermediate/dense
        )

    def call(self, attention_output):
        intermediate_output = self.dense(attention_output)
        return intermediate_output


class BertOutput(tf.keras.layers.Layer):
    # This is the output layer of a full BertLayer (transformer block),
    # combining the output of BertIntermediate with the output of BertAttention (via residual connection)
    def __init__(self, config, name="output", **kwargs):
        super(BertOutput, self).__init__(name=name, **kwargs)
        self.dense = tf.keras.layers.Dense(
            config.hidden_size, 
            kernel_initializer=create_initializer(config.initializer_range), 
            name="dense" # Original scope: bert/encoder/layer_X/output/dense
        )
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-12, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(config.hidden_dropout_prob)

    def call(self, intermediate_output, attention_output, training=None):
        # The residual connection in the original `transformer_model` for this output part is 
        # from the `attention_output`.
        layer_output = self.dense(intermediate_output)
        layer_output = self.dropout(layer_output, training=training)
        layer_output = self.layer_norm(layer_output + attention_output)
        return layer_output


class BertLayer(tf.keras.layers.Layer):
    # This corresponds to one complete Transformer block.
    def __init__(self, config, name="layer", **kwargs):
        super(BertLayer, self).__init__(name=name, **kwargs)
        self.attention = BertAttention(config, name="attention")
        self.intermediate = BertIntermediate(config, name="intermediate")
        self.bert_output_layer = BertOutput(config, name="output") 

    def call(self, hidden_states, attention_mask, training=None):
        attention_output = self.attention(hidden_states, attention_mask, training=training)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.bert_output_layer(intermediate_output, attention_output, training=training)
        return layer_output


class BertModel(tf.keras.Model):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python
    # Already been converted into WordPiece token ids
    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
    token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """
    def __init__(self, config, name="bert", **kwargs):
        super(BertModel, self).__init__(name=name, **kwargs)
        self.config = copy.deepcopy(config)
        self.embeddings_layer = BertEmbeddings(self.config, name="embeddings") # Renamed for clarity
        self.encoder_layers = [BertLayer(self.config, name=f"layer_{i}") 
                               for i in range(self.config.num_hidden_layers)]
        
        self.pooler_dense = tf.keras.layers.Dense(
            config.hidden_size, 
            activation="tanh", 
            kernel_initializer=create_initializer(config.initializer_range),
            name="pooler_dense" # Changed from "pooler/dense"
        )
        # To store outputs if get_ methods are to be used, though direct return from call is preferred.
        self._sequence_output = None
        self._pooled_output = None
        self._all_encoder_layers_outputs = []

    def call(self, input_ids, attention_mask=None, token_type_ids=None, training=None, return_all_encoder_layers=False):
        input_shape = get_shape_list(input_ids, expected_rank=2)
        batch_size = input_shape[0]
        seq_length = input_shape[1]

        if attention_mask is None:
            # Create a default attention mask (all 1s) if none is provided.
            # This mask indicates which tokens to attend to (1) and which not to (0).
            attention_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)
        
        if token_type_ids is None:
            token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

        embedding_output = self.embeddings_layer(input_ids, token_type_ids=token_type_ids, training=training)
        
        # The `create_attention_mask_from_input_mask` function expects `from_tensor` (first arg)
        # to provide shape information, and `to_mask` (second arg) as the actual mask.
        # It produces an additive mask suitable for attention layers.
        # For self-attention, from_tensor (for shape) and to_mask (for mask values) relate to the same sequence.
        # The `embedding_output` provides the correct [batch_size, seq_length, hidden_size] shape.
        # The `attention_mask` is the [batch_size, seq_length] int mask.
        extended_attention_mask = create_attention_mask_from_input_mask(embedding_output, attention_mask)

        self._all_encoder_layers_outputs = [] # Reset for each call
        hidden_states = embedding_output
        for i, layer_module in enumerate(self.encoder_layers):
            hidden_states = layer_module(hidden_states, extended_attention_mask, training=training)
            if return_all_encoder_layers:
                self._all_encoder_layers_outputs.append(hidden_states)
        
        self._sequence_output = hidden_states # Output of the last layer
        if not return_all_encoder_layers:
             # If not returning all, make sure the last layer output is in the list for consistency if get_all_encoder_layers is called.
            self._all_encoder_layers_outputs.append(self._sequence_output)

        # Pooling: take the hidden state corresponding to the first token ([CLS])
        first_token_tensor = self._sequence_output[:, 0] # Slice for the [CLS] token
        self._pooled_output = self.pooler_dense(first_token_tensor)

        if return_all_encoder_layers:
            return self._all_encoder_layers_outputs, self._pooled_output
        else:
            return self._sequence_output, self._pooled_output

    # The get_ methods can still access the stored attributes.
    # However, it's more robust to use the direct outputs from the `call` method.
    def get_pooled_output(self):
        return self._pooled_output 

    def get_sequence_output(self):
        return self._sequence_output 

    def get_all_encoder_layers(self):
        return self._all_encoder_layers_outputs 

    def get_embedding_table(self):
        # Delegate to the embeddings layer instance
        return self.embeddings_layer.get_embedding_table()


def gelu(input_tensor):
    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.
    Original paper: https://arxiv.org/abs/1606.08415

    Args:
        input_tensor: float Tensor to perform activation.

    Returns:
        `input_tensor` with the GELU activation applied.
    """
    cdf = 0.5 * (1.0 + tf.tanh(
        (math.sqrt(2 / math.pi) * (input_tensor + 0.044715 * tf.pow(input_tensor, 3)))))
    return input_tensor * cdf


def get_activation(activation_string):
    """Maps a string to a Python function, e.g., "relu" => `tf.nn.relu`.

    Args:
        activation_string: String name of the activation function.

    Returns:
        A Python function corresponding to the activation function. If
        `activation_string` is None, empty, or "linear", this will return None.
        If `activation_string` is not a string, it will return `activation_string`.

    Raises:
        ValueError: The `activation_string` does not correspond to a known
        activation.
    """
    if not isinstance(activation_string, six.string_types):
        return activation_string

    if not activation_string:
        return None

    act = activation_string.lower()
    if act == "linear":
        return None
    elif act == "relu":
        return tf.nn.relu
    elif act == "gelu":
        return gelu
    elif act == "tanh":
        return tf.tanh
    else:
        raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
    """Compute the union of the current variables and checkpoint variables."""
    assignment_map = {}
    initialized_variable_names = {}

    name_to_variable = collections.OrderedDict()
    for var in tvars:
        name = var.name
        m = re.match("^(.*):\\d+$", name)
        if m is not None:
            name = m.group(1)
        name_to_variable[name] = var

    init_vars = tf.train.list_variables(init_checkpoint)

    assignment_map_tmp = collections.OrderedDict()

    for x in init_vars:
        (name, var) = (x[0], x[1])
        if name not in name_to_variable:
            continue
        
        assignment_map_tmp[name] = name
        initialized_variable_names[name] = 1
        initialized_variable_names[name + ":0"] = 1

    for name_ckpt, name_model in assignment_map_tmp.items():
        assignment_map[name_ckpt] = name_to_variable[name_model]

    return (assignment_map, initialized_variable_names)


def create_initializer(initializer_range=0.02):
    """Creates a `truncated_normal_initializer` with the given range."""
    return tf.keras.initializers.TruncatedNormal(stddev=initializer_range)


def create_attention_mask_from_input_mask(from_tensor, to_mask):
    """Create 3D attention mask from a 2D tensor mask.

    Args:
        from_tensor: 2D or 3D Tensor of shape [batch_size, from_seq_length, ...].
        to_mask: int32 Tensor of shape [batch_size, to_seq_length].

    Returns:
        float Tensor of shape [batch_size, from_seq_length, to_seq_length].
    """
    from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]

    to_shape = get_shape_list(to_mask, expected_rank=2)
    to_seq_length = to_shape[1]

    to_mask = tf.cast(
        tf.reshape(to_mask, [batch_size, 1, to_seq_length]),
        dtype=tf.float32)

    broadcast_ones = tf.ones(
        shape=[batch_size, from_seq_length, 1], dtype=tf.float32)

    mask = broadcast_ones * to_mask
    adder = (1.0 - mask) * -10000.0
    return adder


def get_shape_list(tensor, expected_rank=None, name=None):
    """Returns a list of the shape of tensor, preferring static dimensions.

    Args:
        tensor: A tf.Tensor object to find the shape of.
        expected_rank: (optional) int. The expected rank of `tensor`. If this is
        specified and the `tensor` has a different rank, and exception will be
        thrown.
        name: Optional name of the tensor for the error message.

    Returns:
        A list of dimensions of the shape of tensor. All static dimensions will
        be returned as python integers, and dynamic dimensions will be returned
        as tf.Tensor scalars.
    """
    if name is None:
        if tf.executing_eagerly():
            # In eager mode, tensor.name might not be meaningful or available as in graph mode
            # Provide a default name for context if needed, or rely on the caller to pass one.
            # For now, we'll allow 'name' to remain None if not explicitly passed.
            # If a name is crucial for an operation downstream (e.g. an assert), it should be passed.
            pass # 'name' can remain None
        # else: # Original graph-mode behavior, tensor.name was more reliable
        #    name = tensor.name # This was the problematic line for eager mode

    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name if name else "input_tensor")

    shape = tensor.shape.as_list()

    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)

    if not non_static_indexes:
        return shape
    
    # if not name and any(d is None for d in shape): # TF2 friendly check
    #    tf.compat.v1.logging.warning("Tensor name not provided for shape inference with dynamic dimensions.")
        # Consider if a default name is strictly necessary for tf.shape, usually not.
    
    dyn_shape = tf.shape(input=tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape


def reshape_to_matrix(input_tensor):
    """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
    ndims = input_tensor.shape.ndims
    if ndims < 2:
        raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                         (input_tensor.shape))
    if ndims == 2:
        return input_tensor

    width = input_tensor.shape[-1]
    output_tensor = tf.reshape(input_tensor, [-1, width])
    return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
    """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
    if len(orig_shape_list) == 2:
        return output_tensor

    output_shape = get_shape_list(output_tensor)

    orig_dims = orig_shape_list[0:-1]
    width = output_shape[-1]

    return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
    """Raises an exception if the tensor rank is not of the expected rank.

    Args:
        tensor: A tf.Tensor to check the rank of.
        expected_rank: Python integer or list of integers, expected rank.
        name: Optional name of the tensor for the error message.

    Raises:
        ValueError: If the expected shape doesn't match the actual shape.
    """
    if name is None:
        # In eager mode, tensor.name might not be reliable.
        # Provide a default if no name is given for the error message.
        name = "unnamed_tensor" 
    
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True

    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        scope_name = tf.compat.v1.get_variable_scope().name
        if tf.executing_eagerly():
            scope_name = "eager_execution"
        raise ValueError(
            "For the tensor `%s` in scope `%s`, the actual rank "
            "`%d` (shape = %s) is not equal to the expected rank `%s`" %
            (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))
