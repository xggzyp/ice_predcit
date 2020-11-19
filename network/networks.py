import tensorflow as tf

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from matplotlib import pyplot
from math import sqrt
from sklearn.metrics import mean_squared_error

def create_hparams(args):
    """Create the hparams object for generic training hyperparameters."""
    hparams = tf.contrib.training.HParams(
        gen_num_layers=2,
        dis_num_layers=2,
        gen_rnn_size=128,
        dis_rnn_size=128,
        gen_learning_rate=5e-4,
        dis_learning_rate=5e-3,
        critic_learning_rate=5e-3,
        dis_train_iterations=1,
        gen_learning_rate_decay=1.0,
        gen_full_learning_rate_steps=1e2,
        baseline_decay=0.999999,
        rl_discount_rate=0.9,
        gen_vd_keep_prob=0.5,
        dis_vd_keep_prob=0.5,
        dis_pretrain_learning_rate=5e-3,
        dis_num_filters=128,
        dis_hidden_dim=128,
        gen_nas_keep_prob_0=0.85,
        gen_nas_keep_prob_1=0.55,
        dis_nas_keep_prob_0=0.85,
        dis_nas_keep_prob_1=0.55)
    # Command line flags override any of the preceding hyperparameter values.
    if args.hparams:
        hparams = args.hparams
    return hparams


def fit_network(train_X, train_y, test_X, test_y, args):
    n_features = train_X.shape[2]
    model = Sequential()
    model.add(LSTM(args.hidden, activation='relu', input_shape=(args.n_steps_in, n_features)))
    model.add(RepeatVector(args.n_steps_out))
    model.add(LSTM(args.hidden, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    history = model.fit(train_X, train_y, epochs=args.max_epochs, batch_size=args.batch_size, validation_split=0.2,
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=True)
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()
    # make a prediction
    yhat = model.predict(test_X)

    # invert scaling for forecast
    # inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
    # print("yhat.shape")
    # print(inv_yhat.shape)
    # inv_yhat = scaler.inverse_transform(yhat)
    # inv_yhat = inv_yhat[:, 0]
    # # invert scaling for actual
    # inv_y = scaler.inverse_transform(test_X)
    # inv_y = inv_y[:, 0]
    # calculate RMSE
    test_y=test_y.reshape(test_y.shape[0], n_features*args.n_steps_out)
    yhat=yhat.reshape(test_y.shape[0], n_features*args.n_steps_out)
    rmse = sqrt(mean_squared_error(test_y, yhat))

    print('Test RMSE: %.3f' % rmse, )


def generator(hparams, train_X, train_y, test_X, test_y, args):
    n_features = train_X.shape[2]
    model = Sequential()
    model.add(LSTM(hparams.gen_rnn_size, activation='relu', input_shape=(args.n_steps_in, n_features)))
    model.add(RepeatVector(args.n_steps_out))
    model.add(LSTM(args.hidden, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(n_features)))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    history = model.fit(train_X, train_y, epochs=args.max_epochs, batch_size=args.batch_size, validation_split=0.2,
                        validation_data=(test_X, test_y), verbose=2,
                        shuffle=True)
