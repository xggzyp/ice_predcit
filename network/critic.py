from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Lambda, Concatenate
from keras.layers import LSTM
from keras.layers import Dense
# from keras.layers import

from keras import optimizers
from matplotlib import pyplot
from network.parser import Parser
from network.hparams import create_hparams
from network import attention_utils
import tensorflow as tf
from keras import initializers

args = Parser().get_parser().parse_args()
hparams = create_hparams(args)
# def init_state(shape, name=None):
#     return initializers.Zeros()


def pre_attein(init):
    return attention_utils.prepare_attention(init, num_units=128)


def reshap(x):
    return tf.reshape(x, [-1, 1, 128])


def ex_pen(x):
    return tf.expand_dims(x, axis=1)


def spl(x, index):
    y = x[:, index, :]
    c = tf.expand_dims(y, axis=1)
    return c


initial_state = initializers.Zeros()


def critic_seq2seq_vd_derivative(n_units, n_output, sequence, n_steps_out):  # ,hparams
    def cell_lstm(input_l, state_l):
        decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(
            input_l,
            initial_state=state_l)
        decoder_outputs, state_h, state_c = LSTM(n_units, return_state=True, dropout=0.5,
                                                 recurrent_dropout=0.5)(decoder_lstm1)
        state = [state_h, state_c]
        return decoder_outputs, state

    # print(initial_state[0])
    input_s = Lambda(spl, arguments={'index': 0})(sequence)
    rnn_out, state_gen = cell_lstm(input_s, initial_state)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(rnn_out)
    # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
    decoder_outputs = Lambda(ex_pen)(decoder_outputs)
    out = decoder_outputs
    # out = Reshape((-1,n_units))(rnn_out)
    for i in range(1, n_steps_out):
        input_s = Lambda(spl, arguments={'index': i})(input)
        rnn_out, state_g = cell_lstm(input_s, state_gen)
        state_gen = state_g
        # rnn_out = Reshape((-1,n_units))(rnn_out)
        # print(rnn_out)
        decoder_outputs = decoder_dense(rnn_out)
        # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
        decoder_outputs = Lambda(ex_pen)(decoder_outputs)
        # decoder_outputs=K.expand_dims(decoder_outputs,1)
        out = Concatenate(1)([out, decoder_outputs])

        # pri:nt(decoder_outputs)
        # decoder_outputs=AttentionDecoder(n_units,n_features,decoder_outputs.shape)(decoder_outputs,initial_state[0])
    return out
