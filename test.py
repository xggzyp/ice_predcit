import tensorflow as tf


logit=[0.2,0.15,0.35,0.3]
categorical = tf.distributions.Categorical(logits=logit)

fake = categorical.mode()

fak = categorical.sample((4,2,3))
log_prob = categorical.log_prob(fake)
log_pro = categorical.log_prob(fak)
print(fak)
print(log_prob,log_pro)





















# from keras.layers import Input,initializers
# from keras import backend as K
# import tensorflow as tf
#
#
# init_z = initializers.Zeros()
# inpt=init_z(shape=(128, 1, 9))
# ones = tf.constant(-1,dtype=tf.float32)
# primes_sub_ones = tf.add(inpt, ones)
# print("primes:", primes_sub_ones)
# with tf.Session() as sess:
#     print(sess.run(primes_sub_ones))
# print(inpt)