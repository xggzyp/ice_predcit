from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Lambda, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.layers import LSTM
# from keras.layers import
from network.ice_loader import train_test
from keras import initializers
from keras import optimizers
from matplotlib import pyplot
from network.parser import Parser
from network.hparams import create_hparams
from network import attention_utils
import tensorflow as tf
from network import critic
from keras import backend as K

import matplotlib.pyplot as plt

import sys
from losses import model_losses
from model_units import model_optimization
import numpy as np

args = Parser().get_parser().parse_args()
hparams = create_hparams(args)


def slice(x, index):
    y = x[:, index, :]

    return tf.expand_dims(y, axis=1)

def get_prob(x):
    return tf.log(x)

def get_input(x):
    return x[0]


def pre_attein(init):
    return attention_utils.prepare_attention(init, num_units=128)


def reshap(x):
    return tf.reshape(x, [-1, 1, 128])


def ex_pen(x):
    return tf.expand_dims(x, axis=1)


def g_ex_pen(x):
    x = tf.to_float(x)
    return tf.reshape(x, [-1, 1, 8])


def spl(x, index):
    y = x[:, index, :]
    c = tf.expand_dims(y, axis=1)
    return c


class MKGAN():

    def __init__(self):
        # Input shape
        # self.img_rows = hparams.input
        self.gen_size = hparams.gen_rnn_size
        self.dis_size = hparams.dis_rnn_size
        self.gen_keep_prob = hparams.gen_vd_keep_prob
        self.dir = args.data_dir
        self.batch_size = args.batch_size
        # self.img_cols = 28
        # self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 8
        optimizer = Adam(0.0002, 0.5)
        # self.args = Parser().get_parser().parse_args()
        # self.hparams = create_hparams(self.args)
        self.n_steps_out = 3
        self.n_steps_in = 3
        self.n_output = 8
        self.n_input = 8
        # self.inp = Input(shape=(self.n_steps_in, self.latent_dim), dtype="float32", name="inp")
        # self.sequence=Input(shape=(self.n_steps_out, self.latent_dim), dtype="float32", name="seq")
        # Build and compile the discriminator
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='mean_squared_error',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        # Build the generator

        self.discriminator.trainable = False
        self.gen_dis = self.build_and()
        # The generator takes noise as input and generates imgs

        self.gen_dis.compile(loss='mean_squared_error', optimizer=optimizer)

        # input_d = Input(shape=(1, self.latent_dim))

    def gen_encoder(self, inputs):
        encoder1 = LSTM(self.gen_size, return_state=True, return_sequences=True,
                        dropout=self.gen_keep_prob, recurrent_dropout=self.gen_keep_prob)(
            inputs)
        encoder = LSTM(self.gen_size, return_state=True, return_sequences=True, dropout=self.gen_keep_prob,
                       recurrent_dropout=self.gen_keep_prob)(encoder1)
        encoder_outputs, state_h, state_c = encoder
        encoder_states = [state_h, state_c]
        return encoder_outputs, encoder_states

    def gen_decoder(self, input, encoder_outputs, states, n_output):
        # def init_f(x):
        #     init_z = initializers.Zeros()
        #     return
        #
        # # input=Lambda(init_f)(self.latent_dim)
        # init_z = initializers.Zeros()
        # inpt=init_z(shape=(1, 1, self.latent_dim))
        # inputg=Input(tensor=inpt)
        def cell_lstm(input_l, state_l):
            decoder_lstm1 = LSTM(self.gen_size, return_sequences=True, return_state=True, dropout=0.5,
                                 recurrent_dropout=0.5)(
                input_l,
                initial_state=state_l)
            decoder_output, state_h, state_c = LSTM(self.gen_size, return_state=True, dropout=0.5,
                                                    recurrent_dropout=0.5)(decoder_lstm1)
            # decoder_dense = Dense(n_output, activation='softmax')
            # decoder_outputs = decoder_dense(decoder_output)
            state = [state_h, state_c]
            return decoder_output, state

            # print(initial_state[0])

        rnn_out, state_gen = cell_lstm(input, states)
        # initial_state[0]=Lambda(reshap)(initial_state[0])
        (attention_keys, attention_values) = Lambda(pre_attein)(encoder_outputs)
        attention_score_fn = attention_utils._create_attention_score_fn("attention_keys", self.gen_size, "luong")
        # Attention construction function
        attention_construct_fn = attention_utils._create_attention_construct_fn("attention_score",
                                                                                self.gen_size,
                                                                                attention_score_fn)

        def categori(x):
            logit = x
            categorical = tf.distributions.Categorical(logits=logit)
            print(logit.shape)
            fake = categorical.sample(8)
            log_prob = categorical.log_prob(fake)
            print("wosjofake", fake)
            return ([fake, log_prob])

        def atten(outputs):
            return attention_construct_fn(outputs, attention_keys, attention_values)

        rnn_out = Lambda(atten)(rnn_out)
        if args.gen_train_strategy != 'cross_entory':
            # output, log_prob = Lambda(categori)(rnn_out)
            # out=rnn_out
            decoder_dense = Dense(n_output, activation='sigmoid')
            decoder_outputs = decoder_dense(rnn_out)
            # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
            # decoder_outputs = Lambda(ex_pen)(decoder_outputs)
            # out = decoder_outputs
            log_prob=Lambda(get_prob)(decoder_outputs)
            decoder_outputs = Lambda(ex_pen)(decoder_outputs)
            print("woshideout", decoder_outputs)
            out = decoder_outputs
            log_pro = Lambda(ex_pen)(log_prob)
            # out = Reshape((-1,n_units))(rnn_out)
            for i in range(0, self.n_steps_out - 1):
                rnn_out, state_g = cell_lstm(decoder_outputs, state_gen)
                print("woshiout", rnn_out)
                # print(rnn_out)
                rnn_out = Lambda(atten)(rnn_out)
                # output, log_prob = Lambda(categori)(rnn_out)
                state_gen = state_g
                decoder_dense = Dense(n_output, activation='sigmoid')
                decoder_outputs = decoder_dense(rnn_out)
                # rnn_out = Reshape((-1,n_units))(rnn_out)
                # print(rnn_out)
                log_prob = Lambda(get_prob)(decoder_outputs)
                decoder_outputs = Lambda(ex_pen)(decoder_outputs)
                # decoder_outputs = decoder_dense(rnn_out)
                # # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
                # decoder_outputs = Lambda(ex_pen)(decoder_outputs)
                # decoder_outputs=K.expand_dims(decoder_outputs,1)

                log_prob = Lambda(ex_pen)(log_prob)
                # decoder_outputs=K.expand_dims(decoder_outputs,1)
                out = Concatenate(1)([out, decoder_outputs])
                log_pro = Concatenate(1)([log_pro, log_prob])
            return out, log_pro
        else:
            # ?
            # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
            decoder_outputs = Lambda(ex_pen)(rnn_out)
            out = decoder_outputs
            # out = Reshape((-1,n_units))(rnn_out)
            for i in range(0, self.n_steps_out - 1):
                rnn_out, state_g = cell_lstm(decoder_outputs, state_gen)
                print("woshiout", rnn_out)
                rnn_out = Lambda(atten)(rnn_out)
                state_gen = state_g
                # rnn_out = Reshape((-1,n_units))(rnn_out)
                # print(rnn_out)
                # decoder_outputs = decoder_dense(rnn_out)
                # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
                decoder_outputs = Lambda(ex_pen)(rnn_out)
                # decoder_outputs=K.expand_dims(decoder_outputs,1)
                out = Concatenate(1)([out, decoder_outputs])

            # def init_f(ne):
            #
            #     init_z = initializers.Zeros()
            #     y=init_z(self.batch_size,self.n_steps_out,ne)
            #     return y

        def get_zero():
            d = np.zeros((self.batch_size, self.n_steps_out, self.n_output), dtype="float32")
            return d

        log_pro = Lambda(get_zero)
        # print("woshiout",out)
        return out, log_pro

        # pri:nt(decoder_outputs)
        # decoder_outputs=AttentionDecoder(n_units,n_features,decoder_outputs.shape)(decoder_outputs,initial_state[0])


    def build_generator(self):
        # inputs = Input(shape=(self.n_steps_in, self.latent_dim),dtype="float32",name="gene_input")
        inputs = Input(shape=(self.n_steps_in, self.latent_dim,), dtype="float32", name="inp_gen")
        encoder_outputs, encoder_states = self.gen_encoder(inputs)
        input = Lambda(slice, arguments={'index': -1})(inputs)
        sequence, log_prob = self.gen_decoder(input, encoder_outputs, encoder_states, self.n_output)
        gen_model = Model(inputs, [sequence, log_prob])
        gen_model.summary()
        print("woshigen")
        return gen_model


    def dis_encoder(self, inputs):
        encoder1 = LSTM(self.gen_size, return_state=True, return_sequences=True,
                        dropout=self.gen_keep_prob,
                        recurrent_dropout=self.gen_keep_prob)(
            inputs)
        encoder = LSTM(self.gen_size, return_state=True, return_sequences=True, dropout=self.gen_keep_prob,
                       recurrent_dropout=self.gen_keep_prob)(encoder1)
        encoder_outputs, state_h, state_c = encoder
        encoder_states = [state_h, state_c]

        return encoder_outputs, encoder_states


    def dis_decoder(self, encoder_outputs, states, sequence, n_output):
        def cell_lstm(input_l, state_l):
            decoder_lstm1 = LSTM(self.gen_size, return_sequences=True, return_state=True, dropout=0.5,
                                 recurrent_dropout=0.5)(
                input_l,
                initial_state=state_l)
            decoder_output, state_h, state_c = LSTM(self.gen_size, return_state=True, dropout=0.5,
                                                    recurrent_dropout=0.5)(decoder_lstm1)
            state = [state_h, state_c]
            return decoder_output, state

            # print(initial_state[0])

        input = Lambda(spl, arguments={'index': 0})(sequence)
        rnn_out, state_gen = cell_lstm(input, states)
        # initial_state[0]=Lambda(reshap)(initial_state[0])
        (attention_keys, attention_values) = Lambda(pre_attein)(encoder_outputs)
        attention_score_fn = attention_utils._create_attention_score_fn("attention_keys", self.gen_size, "luong")
        # Attention construction function
        attention_construct_fn = attention_utils._create_attention_construct_fn("attention_score",
                                                                                self.gen_size,
                                                                                attention_score_fn)

        def atten(outputs):
            return attention_construct_fn(outputs, attention_keys, attention_values)

        rnn_out = Lambda(atten)(rnn_out)

        # out=rnn_out

        decoder_dense = Dense(n_output, activation='sigmoid')
        decoder_outputs = decoder_dense(rnn_out)
        decoder_outputs = Lambda(ex_pen)(decoder_outputs)
        # # decoder_outputs = Reshape((-1,1,n_output))(decoder_outputs)
        out = decoder_outputs
        # out = Reshape((-1,n_units))(rnn_out)
        for i in range(1, self.n_steps_out):
            input_s = Lambda(spl, arguments={'index': i})(sequence)
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
        return out


    def build_discriminator(self):
        # inputs= self.generator.inputs
        inputs = Input(shape=(self.n_steps_in, self.latent_dim), dtype="float32", name="inp_dsi")
        sequence = Input(shape=(self.n_steps_out, self.latent_dim), name="sequene")
        # sequence=self.sequence
        dis_encoder_outputs, dis_encoder_states = self.dis_encoder(inputs)
        prediction = self.dis_decoder(dis_encoder_outputs, dis_encoder_states, sequence, self.n_output)
        dis_model = Model([inputs, sequence], prediction)
        dis_model.summary()
        print("dsikhghgg")
        return dis_model


    def build_and(self):
        # inp = Input(shape=(self.n_steps_in, self.latent_dim,), dtype="float32", name="inp")
        inp = self.generator.input
        # inp=Lambda(get_input)(inpw)
        inp_dis, _ = self.discriminator.inputs
        sequence, _ = self.generator(inputs=inp)
        # print(sequence)
        # self.sequence=Input(tensor=sequenc,shape=(self.n_steps_out, self.latent_dim))

        # For the combined model we will only train the generator
        # dis_encoder_outputs, dis_encoder_states = self.dis_encoder(inp)
        # prediction, _ = self.dis_decoder(dis_encoder_outputs, dis_encoder_states, sequence, self.n_output)

        # The discriminator takes generated images as input and determines validity
        # pre = self.discriminator(inputs=[inp,sequence])
        pre = self.discriminator(inputs=[inp_dis, sequence])
        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        and_model = Model([inp, inp_dis], pre)
        # and_model= Model(inp, pre)
        return and_model


    def pretrain_ganerator(self, epochs):
        train_X, train_d, train_y, test_X, test_d, test_y = train_test(self.dir, self.n_steps_in, self.n_steps_out)
        init_z = initializers.Zeros()
        log_pro = init_z(shape=(self.batch_size, self.n_steps_out, self.latent_dim))
        optimizer = Adam(hparams.gen_pretrain_learning_rate, 0.5)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
        for epoch in range(epochs):
            y_pre = self.generator.train_on_batch(train_X, train_y)


    def pretrain_discriminator(self, epochs):
        train_X, train_d, train_y, test_X, test_d, test_y = train_test(self.dir, self.n_steps_in, self.n_steps_out)
        init_z = initializers.Zeros()
        one = init_z(shape=(self.batch_size, self.n_steps_out, self.latent_dim))
        optimizer = Adam(hparams.dis_pretrain_learning_rate)
        self.generator.compile(loss='binary_crossentropy',
                               optimizer=optimizer,
                               metrics=['accuracy'])
        for epoch in range(epochs):
            y_pre = self.discriminator.train_on_batch(train_X, one)


    def train(self, epochs, save_interval=50):
        batch_size = self.batch_size
        # Load the dataset
        train_X, train_d, train_y, test_X, test_d, test_y = train_test(self.dir, self.n_steps_in, self.n_steps_out)

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        train_d = np.expand_dims(train_d, axis=1)
        test_d = np.expand_dims(test_d, axis=1)

        # Adversarial ground truths
        valid = np.ones((batch_size, self.n_steps_out, self.n_output), dtype="float32")
        fake = np.zeros((batch_size, self.n_steps_out, self.n_output), dtype="float32")

        # history=self.gen_dis.fit([train_X,train_X],valid,batch_size=16,epochs=30,verbose=2)

        for epoch in range(epochs):
            # ---------------------
            #  Train Discriminator
            # ---------------------

            idx = np.random.randint(0, train_X.shape[0], batch_size)
            imgs = train_X[idx]
            imgs_d = train_d[idx]
            real_sequence = train_y[idx]

            # # Sample noise and generate a batch of new images
            # noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            # gen_imgs = self.generator.predict(noise)
            fake_sequence,fake_log_probs = self.generator.predict(imgs, batch_size=128)



            est_state_values = critic.critic_seq2seq_vd_derivative(
                128, self.n_output, fake_sequence, self.n_steps_out
                )
            [
                fake_RL_loss, fake_log_probs, fake_rewards, fake_advantages,
                fake_baselines, fake_averages_op, critic_loss, cumulative_rewards
            ] = model_losses.calculate_reinforce_objective(
                hparams, fake_log_probs, fake_sequence, est_state_values)
            ##  Generator Train Op.
            # 1.  Cross-Entropy.
            if args.gen_training_strategy == 'cross_entropy':
                gen_loss = tf.reduce_mean(fake_cross_entropy_losses)
                [gen_train_op, gen_grads,
                 gen_vars] = model_optimization.create_gen_train_op(
                    hparams, learning_rate, gen_loss, global_step, mode='MINIMIZE')

            # 2.  GAN (REINFORCE)
            elif FLAGS.gen_training_strategy == 'reinforce':
                gen_loss = fake_RL_loss
                [gen_train_op, gen_grads,
                 gen_vars] = model_optimization.create_reinforce_gen_train_op(
                    hparams, learning_rate, gen_loss, fake_averages_op, global_step)

            else:
                raise NotImplementedError

            ## Discriminator Train Op.
            dis_train_op, dis_grads, dis_vars = model_optimization.create_dis_train_op(
                hparams, dis_loss, global_step)

            ## Critic Train Op.
            if critic_loss is not None:
                [critic_train_op, _, _] = model_optimization.create_critic_train_op(
                    hparams, critic_loss, global_step)
                dis_train_op = tf.group(dis_train_op, critic_train_op)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch([imgs, real_sequence], valid)
            d_loss_fake = self.discriminator.train_on_batch([imgs, fake_sequence], fake)
            print(d_loss_real, d_loss_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print("开始训练，，，，，，，，，，，，，，，，，，，，，，，，，，，，")
            print(d_loss)
            # ---------------------
            #  Train Generator
            # ---------------------
            print(imgs.shape)
            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.gen_dis.train_on_batch([imgs, imgs], valid)
            gradients = K.gradients(self.gen_dis.output,
                                    self.gen_dis.input)  # Gradient of output wrt the input of the model (Tensor)
            print("woshitidu{}".format(gradients))
            # g_loss = self.gen_dis.train_on_batch(imgs, valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            # if epoch % save_interval == 0:
            #     print()


    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("images/mnist_%d.png" % epoch)
        plt.close()


if __name__ == '__main__':
    mkgan = MKGAN()
    mkgan.train(epochs=100)
