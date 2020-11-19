from keras.layers import Input,initializers
from keras import backend as K
import tensorflow as tf


init_z = initializers.Zeros()
inpt=init_z(shape=(128, 1, 9))
ones = tf.constant(-1,dtype=tf.float32)
primes_sub_ones = tf.add(inpt, ones)
print("primes:", primes_sub_ones)
with tf.Session() as sess:
    print(sess.run(primes_sub_ones))
print(inpt)