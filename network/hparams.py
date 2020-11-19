import tensorflow as tf


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
