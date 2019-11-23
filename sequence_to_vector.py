# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """
    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """
    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._dropout = dropout
        self._num_layers = num_layers
        self._input_dim = input_dim
        layers = []
        for i in range(self._num_layers):
            # Define the DAN layers
            layers.append(tf.keras.layers.Dense(input_dim, activation='relu'))
        self._layers = layers
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        shapes = vector_sequence.get_shape().as_list()
        list_tensors = []

        # DROPOUT_CALCULATION_START
        # if in training mode, apply dropout
        if training:
            # This call calculates a uniform distribution with min_value = 0 and max_value = 1
            uniform = tf.random.uniform( [shapes[1]], 0,1)
            uniform_list = uniform.numpy()
            dropout_list = []
            for i in range(len(uniform_list)):
                # if probability is greater than dropout probability, then assign 1, else assign 0
                if uniform_list[i]>self._dropout:
                    dropout_list.append(1)
                else:
                    dropout_list.append(0)

            dropout_list_input_dim = [ dropout_list for i in range(shapes[2])]
            dropout_tf = tf.convert_to_tensor(dropout_list_input_dim, dtype=tf.float32)

            do_list = []
            # reshape the input vector_sequence to appropriate shape for applying dropout
            vector_sequence_for_dropout = tf.reshape(vector_sequence, [shapes[0], shapes[2], shapes[1]])
            for i in range(shapes[0]):
                    # apply dropout mask
                    do_list.append(tf.math.multiply(vector_sequence_for_dropout[i], dropout_tf))

            do_tensor = tf.convert_to_tensor(do_list)
        # DROPOUT_CALCULATION_END

        else:
            # do not apply dropout if not in training mode
            do_tensor = vector_sequence

        # reshape the tensor to the appropriate shape, so that sequence_mask can be applied
        vector_sequence_reshaped = tf.reshape(do_tensor, [shapes[2], shapes[0], shapes[1]])
        for i in range(self._input_dim):
            # apply sequence_mask
             list_tensors.append(tf.math.multiply(vector_sequence_reshaped[i], sequence_mask ))

        vector_sequence_converted = tf.convert_to_tensor(list_tensors, dtype=tf.float32)
        vector_sequence_converted_reshaped = tf.reshape(vector_sequence_converted,[shapes[0],shapes[1],shapes[2]])

        # take the average of words for each sentence in a batch ( the first step in a Deep Averaging Network )
        reducedMean = tf.math.reduce_mean(vector_sequence_converted_reshaped,1)
        layer_output = reducedMean
        lrTensor = []
        for i in range(self._num_layers):
            # pass the input to the ' ith' layer and the output of the 'ith' layer to the '(i+1)th' layer in the DAN.
            layer_output = self._layers[i](layer_output)
            # extract layer_representations at each layer and append it to a temp list
            lrTensor.append(layer_output)

        # put the last layer to the combined_vector
        combined_vector = layer_output
        # convert the layer_representations list to a tensor, that can be passed as a part of the output
        layer_representations = tf.convert_to_tensor(lrTensor)
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """
    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._num_layers = num_layers
        gru_layers = []
        for i in range(num_layers):
            # Define the GRU layers
            gru_layers.append(tf.keras.layers.GRU(input_dim))
        self._gru_layers= gru_layers
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        lo = []
        # TODO(students): start
        for i in range(self._num_layers):
            # extract the GRU layer_reprsentations at each layer and append it to a list
            layer_output = self._gru_layers[i](vector_sequence, mask=sequence_mask)
            lo.append(layer_output)

        # convert the layer_representations list to a tensor, which needs to be returned as a part of the output
        layer_representations = tf.convert_to_tensor(lo)
        # Put the last layer_representation to the combined_vector, which is one of the output
        combined_vector = layer_output
        # TODO(students): end
        return {"combined_vector": combined_vector,
                "layer_representations": layer_representations}
