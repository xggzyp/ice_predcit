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


# generate a sequence of random integers
def generate_sequence(length, n_unique):
    return [randint(1, n_unique - 1) for _ in range(length)] #产生1-50的随机整数


# prepare data for the LSTM
def get_dataset(n_in, n_out, cardinality, n_samples):
    X1, X2, y = list(), list(), list()
    for _ in range(n_samples):
        # generate source sequence
        source = generate_sequence(n_in, cardinality)
        # define padded target sequence
        target = source[:n_out]
        target.reverse() #颠倒元素中的顺序
        # create padded input target sequence
        target_in = [0] + target[:-1]
        print("X1是{}/nX2shi{}/nY是{}/n  ".format(source[:5], target[:5], target_in[:5]))
        # encode
        src_encoded = to_categorical([source], num_classes=cardinality)
        tar_encoded = to_categorical([target], num_classes=cardinality)
        tar2_encoded = to_categorical([target_in], num_classes=cardinality)

        # store
        X1.append(src_encoded)
        X2.append(tar2_encoded)
        y.append(tar_encoded)
    X1 = array(X1).reshape((n_samples, n_in, cardinality))
    X2 = array(X2).reshape((n_samples, n_out, cardinality))
    y = array(y).reshape((n_samples, n_out, cardinality))
    return array(X1), array(X2), array(y)


# def slice(x,index):
#     return x[index]
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


def decoder_lstm(n_units, n_output, encoder_outputs, input, initial_state, parser, n_steps_out):  # ,hparams
    def cell_lstm(input_l, state_l):
        decoder_lstm1 = LSTM(n_units, return_sequences=True, return_state=True, dropout=0.5, recurrent_dropout=0.5)(
            input_l,
            initial_state=state_l)
        decoder_outputs, state_h, state_c = LSTM(n_units, return_state=True, dropout=0.5,
                                                 recurrent_dropout=0.5)(decoder_lstm1)
        state = [state_h, state_c]
        return decoder_outputs, state

    # print(initial_state[0])
    input_s = Lambda(spl, arguments={'index': 0})(input)
    rnn_out, state_gen = cell_lstm(input_s, initial_state)
    # initial_state[0]=Lambda(reshap)(initial_state[0])
    (attention_keys, attention_values) = Lambda(pre_attein)(encoder_outputs)
    attention_score_fn = attention_utils._create_attention_score_fn("attention_keys", n_units, "luong")
    # Attention construction function
    attention_construct_fn = attention_utils._create_attention_construct_fn("attention_score",
                                                                            n_units, attention_score_fn)
    attention_option = parser.attention_option

    def atten(outputs):
        return attention_construct_fn(outputs, attention_keys, attention_values)

    rnn_out = Lambda(atten)(rnn_out)
    # out=rnn_out
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(rnn_out)
    # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
    decoder_outputs = Lambda(ex_pen)(decoder_outputs)
    out = decoder_outputs
    # out = Reshape((-1,n_units))(rnn_out)
    for i in range(1, n_steps_out):
        input_s = Lambda(spl, arguments={'index': i})(input)
        rnn_out, state_g = cell_lstm(input_s, state_gen)
        rnn_out = Lambda(atten)(rnn_out)
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
    return out, state_gen


# state_h1 = Lambda(slice, output_shape=(1,), arguments={'index': 0})(initial_state)
# state_h1=Layer(state_he)
# print(array(initial_state).shape)


def concet_i(x, y):
    return [x, y]


def dis_decoder_lstm(n_units, n_output, encoder_outputs,sequence, initial_state, parser, n_steps_out):  # ,hparams
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
    # initial_state[0]=Lambda(reshap)(initial_state[0])
    (attention_keys, attention_values) = Lambda(pre_attein)(encoder_outputs)
    attention_score_fn = attention_utils._create_attention_score_fn("attention_keys", n_units, "luong")
    # Attention construction function
    attention_construct_fn = attention_utils._create_attention_construct_fn("attention_score",
                                                                            n_units, attention_score_fn)
    attention_option = parser.attention_option

    def atten(outputs):
        return attention_construct_fn(outputs, attention_keys, attention_values)

    rnn_out = Lambda(atten)(rnn_out)
    # out=rnn_out
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(rnn_out)
    # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
    decoder_outputs = Lambda(ex_pen)(decoder_outputs)
    out = decoder_outputs
    # out = Reshape((-1,n_units))(rnn_out)
    for i in range(1, n_steps_out):
        input_s = Lambda(spl, arguments={'index': i})(input)
        rnn_out, state_g = cell_lstm(input_s, state_gen)
        rnn_out = Lambda(atten)(rnn_out)
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
    return out, state_gen
def model_dis(n_input, n_output,n_units,parser,inputs,
              targets_present,
              sequence,
              is_training, ):
    encoder_inputs = Input(shape=(None, n_input))
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(
        encoder_inputs)
    encoder = LSTM(n_units, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(encoder1)
    encoder_outputs, state_h, state_c = encoder
    encoder_states = [state_h, state_c]
    #dis_decoder
    sequence = Input(shape=(None, n_output))

    decoder_outputs, _ = dis_decoder_lstm(n_units, n_output, encoder_outputs,sequence, encoder_states, parser,
                                      n_steps_out)

    print(decoder_outputs)
    # mddel_input = Lambda(concet_i(encoder_inputs,decoder_inputs))
    # define inference encoder
    # input=K.concatenate([encoder_inputs, decoder_inputs],axis=0)
    encoder_model = Model(encoder_inputs, encoder_states)
    # print(encoder_inputs, encoder_states, decoder_outputs)
    model_dis = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    return



# returns train, inference_encoder and inference_decoder models
def define_models(n_input, n_output, n_units, parser, n_steps_out):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder1 = LSTM(n_units, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(
        encoder_inputs)
    encoder = LSTM(n_units, return_state=True, return_sequences=True, dropout=0.5, recurrent_dropout=0.5)(encoder1)
    encoder_outputs, state_h, state_c = encoder
    encoder_states = [state_h, state_c]
    # define training decoder
    print("woshostate_h{}\nwoshiostate_c{}".format(state_h, state_c))
    decoder_inputs = Input(shape=(None, n_output))

    decoder_outputs, _ = decoder_lstm(n_units, n_output, encoder_outputs, decoder_inputs, encoder_states, parser,
                                      n_steps_out)

    print(decoder_outputs)
    # mddel_input = Lambda(concet_i(encoder_inputs,decoder_inputs))
    # define inference encoder
    # input=K.concatenate([encoder_inputs, decoder_inputs],axis=0)
    encoder_model = Model(encoder_inputs, encoder_states)
    # print(encoder_inputs, encoder_states, decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, decoder_states = decoder_lstm(n_units, n_output, encoder_outputs, decoder_inputs,
                                                   decoder_states_inputs, parser, n_steps_out)
    # decoder_states = [state_h, state_c]
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model


# generate target given source sequence
def predict_sequence(infenc, infdec, source, n_steps, cardinality):
    # encode
    print(source.shape)
    print()
    state = infenc.predict(source)

    # start of sequence input
    target_seq = array([0.0 for _ in range(cardinality)]).reshape((1, 1, cardinality))
    # collect predictions
    output = list()

    yhat, _ = infdec.predict([target_seq])

    return array(output)


# decode a one hot encoded string
def one_hot_decode(encoded_seq):
    return [argmax(vector) for vector in encoded_seq]


"""之后可删，纳入到总得train.py中"""
args = Parser().get_parser().parse_args()
hparams = create_hparams(args)

# configure problem
n_features = 50 + 1
n_steps_in = 4
n_steps_out = 4
# define model
train, infenc, infdec = define_models(n_features, n_features, 128, args, n_steps_out)
adam = optimizers.Adam(lr=0.001, decay=1e-6)

train.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
# generate training dataset
X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 8000)

print(X1.shape, X2.shape, y.shape)
# train model
# X1=X1.reshape((100000,n_steps_in,n_features))
# X2=X2.reshape((100000,n_steps_out,n_features))
# y=y.reshape((100000,n_steps_out,n_features))
X_test1, X_test2, y_test = get_dataset(n_steps_in, n_steps_out, n_features, 2000)
print(X_test1.shape, X_test2.shape, y_test.shape)

history = train.fit([X1, X2], y, epochs=30, verbose=2, batch_size=128,
                    validation_data=([X_test1, X_test2], y_test))  #
pyplot.plot(history.history['acc'], label='train')
pyplot.plot(history.history['val_acc'], label='test')
pyplot.legend()
pyplot.show()
# evaluate LSTM
total, correct = 100, 0
for _ in range(total):
    # tf.reset_default_graph()

    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    # target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    target = train.predict([X1, X2], batch_size=1)
    if array_equal(one_hot_decode(y[0]), one_hot_decode(target[0])):
        correct += 1
print('Accuracy: %.2f%%' % (float(correct) / float(total) * 100.0))
# spot check some examples
for _ in range(10):
    X1, X2, y = get_dataset(n_steps_in, n_steps_out, n_features, 1)
    # print(X1)
    # target = predict_sequence(infenc, infdec, X1, n_steps_out, n_features)
    target = train.predict([X1, X2], batch_size=1)
    print('X=%s y=%s, yhat=%s' % (one_hot_decode(X1[0]), one_hot_decode(y[0]), one_hot_decode(target[0])))
