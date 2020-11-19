from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import numpy as np
from six.moves import xrange
import tensorflow as tf

# Useful for REINFORCE baseline.
from losses import losses

FLAGS = tf.app.flags.FLAGS
args = Parser().get_parser().parse_args()
hparams = create_hparams(args)

def create_dis_loss(fake_predictions, real_predictions, targets_present):
    """Compute Discriminator loss across real/fake."""

    real_labels = tf.ones([FLAGS.batch_size, FLAGS.sequence_length])
    dis_loss_real = tf.losses.sigmoid_cross_entropy(
        real_labels, real_predictions)
    dis_loss_fake = tf.losses.sigmoid_cross_entropy(
        targets_present, fake_predictions)

    dis_loss = (dis_loss_fake + dis_loss_real) / 2.
    return dis_loss, dis_loss_fake, dis_loss_real


def create_critic_loss(cumulative_rewards, estimated_values, present):
    """Compute Critic loss in estimating the value function.  This should be an
    estimate only for the missing elements."""
    missing = tf.cast(present, tf.int32)
    missing = 1 - missing
    missing = tf.cast(missing, tf.bool)

    loss = tf.losses.mean_squared_error(
        labels=cumulative_rewards, predictions=estimated_values, weights=missing)
    return loss
