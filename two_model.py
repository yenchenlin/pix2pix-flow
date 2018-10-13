import tensorflow as tf

import tfops as Z
import two_optim as optim
import numpy as np
import horovod.tensorflow as hvd
from tensorflow.contrib.framework.python.ops import add_arg_scope


'''
f_loss: function with as input the (x,y,reuse=False), and as output a list/tuple whose first element is the loss.
'''


def abstract_model_xy(sess, hps, feeds, train_iterators, test_iterators, data_inits, lr, f_loss):

    # == Create class with static fields and methods
    class m(object):
        pass
    m.sess = sess
    m.feeds = feeds
    m.lr = lr

    # === Loss and optimizer
    if hps.joint_train and not hps.inference:
        loss_train_A, stats_train_A, eps_flatten_A, loss_train_B, stats_train_B, eps_flatten_B = f_loss(train_iterators, is_training=True)
    else:
        loss_train_A, stats_train_A, loss_train_B, stats_train_B = f_loss(train_iterators, is_training=True)

    all_params = tf.trainable_variables()

    # Get train data op
    def get_train_data():
        x_A, y_A = train_iterators['A']()
        x_B, y_B = train_iterators['B']()
        return x_A, y_A, x_B, y_B
    m.get_train_data = get_train_data

    # A
    with tf.variable_scope('optim_A'):
        params_A = [param for param in all_params if 'A/' in param.name]
        if hps.gradient_checkpointing == 1:
            from memory_saving_gradients import gradients
            gs_A = gradients(loss_train_A, params_A)
        else:
            gs_A = tf.gradients(loss_train_A, params_A)
        m.optimizer_A = optim.Optimizer()
        train_op_A, polyak_swap_op_A, ema_A = m.optimizer_A.adamax(
            params_A, gs_A, alpha=lr, hps=hps)
        if hps.direct_iterator:
            m.train_A = lambda _lr: sess.run([train_op_A, stats_train_A], {lr: _lr})[1]
        else:
            def _train_A(_lr, _x_A, _y_A, _x_B, _y_B):
                return sess.run([train_op_A, stats_train_A], {feeds['x_A']: _x_A,
                                                              feeds['y_A']: _y_A,
                                                              feeds['x_B']: _x_B,
                                                              feeds['y_B']: _y_B,
                                                              lr: _lr})[1]
            m.train_A = _train_A
        m.polyak_swap_A = lambda: sess.run(polyak_swap_op_A)
    # B
    with tf.variable_scope('optim_B'):
        params_B = [param for param in all_params if 'B/' in param.name]
        if hps.gradient_checkpointing == 1:
            from memory_saving_gradients import gradients
            gs_B = gradients(loss_train_B, params_B)
        else:
            gs_B = tf.gradients(loss_train_B, params_B)
        m.optimizer_B = optim.Optimizer()
        train_op_B, polyak_swap_op_B, ema_B = m.optimizer_B.adamax(
            params_B, gs_B, alpha=lr, hps=hps)
        if hps.direct_iterator:
            m.train_B = lambda _lr: sess.run([train_op_B, stats_train_B], {lr: _lr})[1]
        else:
            def _train_B(_lr, _x_A, _y_A, _x_B, _y_B):
                return sess.run([train_op_B, stats_train_B], {feeds['x_A']: _x_A,
                                                              feeds['y_A']: _y_A,
                                                              feeds['x_B']: _x_B,
                                                              feeds['y_B']: _y_B,
                                                              lr: _lr})[1]
            m.train_B = _train_B
        m.polyak_swap_B = lambda: sess.run(polyak_swap_op_B)

    # === Testing
    loss_test_A, stats_test_A, loss_test_B, stats_test_B = f_loss(test_iterators, False, reuse=True)
    if hps.direct_iterator:
        m.test_A = lambda: sess.run(stats_test_A)
        m.test_B = lambda: sess.run(stats_test_B)
    else:
        # Get test data op
        def get_test_data():
            x_A, y_A = test_iterators['A']()
            x_B, y_B = test_iterators['B']()
            return x_A, y_A, x_B, y_B
        m.get_test_data = get_test_data

        def _test_A(_x_A, _y_A, _x_B, _y_B):
            return sess.run(stats_test_A, {feeds['x_A']: _x_A,
                                           feeds['y_A']: _y_A,
                                           feeds['x_B']: _x_B,
                                           feeds['y_B']: _y_B})
        def _test_B(_x_A, _y_A, _x_B, _y_B):
            return sess.run(stats_test_B, {feeds['x_A']: _x_A,
                                           feeds['y_A']: _y_A,
                                           feeds['x_B']: _x_B,
                                           feeds['y_B']: _y_B})
        m.test_A = _test_A
        m.test_B = _test_B

    # === Saving and restoring
    with tf.variable_scope('saver_A'):
        saver_A = tf.train.Saver()
        saver_ema_A = tf.train.Saver(ema_A.variables_to_restore())
        m.save_ema_A = lambda path_A: saver_ema_A.save(
            sess, path_A, write_meta_graph=False)
        m.save_A = lambda path_A: saver_A.save(sess, path_A, write_meta_graph=False)
        m.restore_A = lambda path_A: saver_A.restore(sess, path_A)

    with tf.variable_scope('saver_B'):
        saver_B = tf.train.Saver()
        saver_ema_B = tf.train.Saver(ema_B.variables_to_restore())
        m.save_ema_B = lambda path_B: saver_ema_B.save(
            sess, path_B, write_meta_graph=False)
        m.save_B = lambda path_B: saver_B.save(sess, path_B, write_meta_graph=False)
        m.restore_B = lambda path_B: saver_B.restore(sess, path_B)
        print("After saver")

    # === Initialize the parameters
    if hps.restore_path_A != '':
        m.restore_A(hps.restore_path_A)
    if hps.restore_path_B != '':
        m.restore_B(hps.restore_path_B)
    if hps.restore_path_A == '' and hps.restore_path_B == '':
        with Z.arg_scope([Z.get_variable_ddi, Z.actnorm], init=True):
            results_init = f_loss(None, False, reuse=True)

        all_params = tf.global_variables()
        params_A = [param for param in all_params if 'A/' in param.name]
        params_B = [param for param in all_params if 'B/' in param.name]
        sess.run(tf.variables_initializer(params_A))
        sess.run(tf.variables_initializer(params_B))
        feeds_dict = {feeds['x_A']: data_inits['A']['x'],
                      feeds['y_A']: data_inits['A']['y'],
                      feeds['x_B']: data_inits['B']['x'],
                      feeds['y_B']: data_inits['B']['y']}
        sess.run(results_init, feeds_dict)
    sess.run(hvd.broadcast_global_variables(0))

    return m


def codec(hps):

    def encoder(z, objective):
        eps = []
        for i in range(hps.n_levels):
            z, objective = revnet2d(str(i), z, objective, hps)
            if i < hps.n_levels-1:
                z, objective, _eps = split2d("pool"+str(i), z, objective=objective)
                eps.append(_eps)
        return z, objective, eps

    def decoder(z, eps=[None]*hps.n_levels, eps_std=None):
        for i in reversed(range(hps.n_levels)):
            if i < hps.n_levels-1:
                z = split2d_reverse("pool"+str(i), z, eps=eps[i], eps_std=eps_std)
            z, _ = revnet2d(str(i), z, 0, hps, reverse=True)

        return z

    return encoder, decoder


def prior(name, y_onehot, hps):

    with tf.variable_scope(name):
        n_z = hps.top_shape[-1]

        h = tf.zeros([tf.shape(y_onehot)[0]]+hps.top_shape[:2]+[2*n_z])
        if hps.learntop:
            h = Z.conv2d_zeros('p', h, 2*n_z)
        if hps.ycond:
            h += tf.reshape(Z.linear_zeros("y_emb", y_onehot,
                                           2*n_z), [-1, 1, 1, 2 * n_z])

        pz = Z.gaussian_diag(h[:, :, :, :n_z], h[:, :, :, n_z:])

    def logp(z1):
        objective = pz.logp(z1)
        return objective

    def sample(eps=None, eps_std=None):
        if eps is not None:
            # Already sampled eps. Don't use eps_std
            z = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z = pz.sample

        return z

    def eps(z1):
        return pz.get_eps(z1)

    return logp, sample, eps


def model(sess, hps, train_iterators, test_iterators, data_inits):

    # Only for decoding/init, rest use iterators directly
    with tf.name_scope('input'):
        # Input A
        X_A = tf.placeholder(
            tf.uint8, [None, hps.image_size, hps.image_size, 3], name='image_A')
        Y_A = tf.placeholder(tf.int32, [None], name='label_A')
        # Input B
        X_B = tf.placeholder(
            tf.uint8, [None, hps.image_size, hps.image_size, 3], name='image_B')
        Y_B = tf.placeholder(tf.int32, [None], name='label_B')
        # learning rate
        lr = tf.placeholder(tf.float32, None, name='learning_rate')

    with tf.variable_scope('A'):
        encoder_A, decoder_A = codec(hps)
    with tf.variable_scope('B'):
        encoder_B, decoder_B = codec(hps)
    hps.n_bins = 2. ** hps.n_bits_x

    def preprocess(x):
        x = tf.cast(x, 'float32')
        if hps.n_bits_x < 8:
            x = tf.floor(x / 2 ** (8 - hps.n_bits_x))
        x = x / hps.n_bins - .5
        return x

    def postprocess(x):
        return tf.cast(tf.clip_by_value(tf.floor((x + .5)*hps.n_bins)*(256./hps.n_bins), 0, 255), 'uint8')

    def _f_loss(x_A, y_A, x_B, y_B, is_training, reuse=False):
        with tf.variable_scope('model_A', reuse=reuse):
            y_onehot_A = tf.cast(tf.one_hot(y_A, hps.n_y, 1, 0), 'float32')

            # Discrete -> Continuous
            objective_A = tf.zeros_like(x_A, dtype='float32')[:, 0, 0, 0]
            z_A = preprocess(x_A)
            z_A = z_A + tf.random_uniform(tf.shape(z_A), 0, 1./hps.n_bins)
            objective_A += - np.log(hps.n_bins) * np.prod(Z.int_shape(z_A)[1:])

            # Encode
            z_A = Z.squeeze2d(z_A, 2)  # > 16x16x12
            z_A, objective_A, eps_A = encoder_A(z_A, objective_A)

            # Prior
            hps.top_shape = Z.int_shape(z_A)[1:]
            logp_A, _, _eps_A = prior("prior", y_onehot_A, hps)
            objective_A += logp_A(z_A)

            # Note that we learn the top layer so need to process z
            z_A = _eps_A(z_A)
            eps_A.append(z_A)

            # Loss of eps and flatten latent code from another model
            eps_flatten_A = tf.concat([tf.contrib.layers.flatten(e) for e in eps_A], axis=-1)

        with tf.variable_scope('model_B', reuse=reuse):
            y_onehot_B = tf.cast(tf.one_hot(y_B, hps.n_y, 1, 0), 'float32')

            # Discrete -> Continuous
            objective_B = tf.zeros_like(x_B, dtype='float32')[:, 0, 0, 0]
            z_B = preprocess(x_B)
            z_B = z_B + tf.random_uniform(tf.shape(z_B), 0, 1./hps.n_bins)
            objective_B += - np.log(hps.n_bins) * np.prod(Z.int_shape(z_B)[1:])

            # Encode
            z_B = Z.squeeze2d(z_B, 2)  # > 16x16x12
            z_B, objective_B, eps_B = encoder_B(z_B, objective_B)

            # Prior
            hps.top_shape = Z.int_shape(z_B)[1:]
            logp_B, _, _eps_B = prior("prior", y_onehot_B, hps)
            objective_B += logp_B(z_B)

            # Note that we learn the top layer so need to process z
            z_B = _eps_B(z_B)
            eps_B.append(z_B)

            # Loss of eps and flatten latent code from another model
            eps_flatten_B = tf.concat([tf.contrib.layers.flatten(e) for e in eps_B], axis=-1)

        code_loss = 0.0
        code_shapes = [[16, 16, 6], [8, 8, 12], [4, 4, 48]]
        if hps.code_loss_type == 'B_all':
            """ Decode the code from another model and compute L2 loss
                at pixel level
            """
            def unflatten_code(fcode, code_shapes):
                index = 0
                code = []
                bs = tf.shape(fcode)[0]
                # bs = hps.local_batch_train
                for shape in code_shapes:
                    code.append(tf.reshape(fcode[:, index:index+np.prod(shape)],
                                           tf.convert_to_tensor([bs] + shape)))
                    index += np.prod(shape)
                return code

            code_others = unflatten_code(eps_flatten_A, code_shapes)
            # code_others[-1] is z, and code_others[:-1] is eps
            _, sample, _ = prior("prior", y_onehot_B, hps)
            code_last_others = sample(eps=code_others[-1])
            code_decoded_others = decoder_B(code_last_others, code_others[:-1])
            code_decoded = Z.unsqueeze2d(code_decoded_others, 2)
            x_B_pred = postprocess(code_decoded)
            code_loss = tf.reduce_mean(tf.squared_difference(tf.cast(x_B_pred, tf.float32), tf.cast(x_B, tf.float32)))
        elif hps.code_loss_type == 'code_all':
            code_loss = tf.reduce_mean(
                tf.squared_difference(eps_flatten_A, eps_flatten_B))
        elif hps.code_loss_type == 'code_last':
            dim = np.prod(code_shapes[-1])
            code_loss = tf.reduce_mean(tf.squared_difference(eps_flatten_A[:, -dim:], eps_flatten_B[:, -dim:]))
        else:
            raise NotImplementedError()

        with tf.variable_scope('model_A', reuse=True):
            # Generative loss
            nobj_A = - objective_A
            bits_x_A = nobj_A / (np.log(2.) * int(x_A.get_shape()[1]) * int(
                x_A.get_shape()[2]) * int(x_A.get_shape()[3]))  # bits per subpixel
            bits_y_A = tf.zeros_like(bits_x_A)
            classification_error_A = tf.ones_like(bits_x_A)

        with tf.variable_scope('model_B', reuse=True):
            # Generative loss
            nobj_B = - objective_B
            bits_x_B = nobj_B / (np.log(2.) * int(x_B.get_shape()[1]) * int(
                x_B.get_shape()[2]) * int(x_B.get_shape()[3]))  # bits per subpixel
            bits_y_B = tf.zeros_like(bits_x_B)
            classification_error_B = tf.ones_like(bits_x_B)

        return bits_x_A, bits_y_A, classification_error_A, eps_flatten_A, bits_x_B, bits_y_B, classification_error_B, eps_flatten_B, code_loss

    def f_loss(iterators, is_training, reuse=False):
        if hps.direct_iterator and iterators is not None:
            raise NotImplementedError()
        else:
            x_A, y_A, x_B, y_B = X_A, Y_A, X_B, Y_B

        bits_x_A, bits_y_A, pred_loss_A, eps_flatten_A, bits_x_B, bits_y_B, pred_loss_B, eps_flatten_B, code_loss = _f_loss(x_A, y_A, x_B, y_B, is_training, reuse)
        local_loss_A = hps.mle_loss_scale * bits_x_A + hps.weight_y * bits_y_A
        local_loss_B = hps.mle_loss_scale * bits_x_B + hps.weight_y * bits_y_B
        # Add code difference loss
        if hps.joint_train:
            local_loss_A += hps.code_loss_scale * code_loss
            local_loss_B += hps.code_loss_scale * code_loss

        stats_A = [local_loss_A, bits_x_A, bits_y_A, pred_loss_A, code_loss]
        stats_B = [local_loss_B, bits_x_B, bits_y_B, pred_loss_B, code_loss]
        global_stats_A = Z.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats_A]))
        global_stats_B = Z.allreduce_mean(
            tf.stack([tf.reduce_mean(i) for i in stats_B]))

        if hps.joint_train and is_training:
            return tf.reduce_mean(local_loss_A), global_stats_A, eps_flatten_A, tf.reduce_mean(local_loss_B), global_stats_B, eps_flatten_B
        else:
            return tf.reduce_mean(local_loss_A), global_stats_A, tf.reduce_mean(local_loss_B), global_stats_B

    feeds = {'x_A': X_A, 'y_A': Y_A, 'x_B': X_B, 'y_B': Y_B}
    m = abstract_model_xy(sess, hps, feeds, train_iterators,
                          test_iterators, data_inits, lr, f_loss)

    # # === Sampling function
    def f_sample(y_A, y_B, eps_std):
        with tf.variable_scope('model_A', reuse=True):
            y_onehot_A = tf.cast(tf.one_hot(y_A, hps.n_y, 1, 0), 'float32')

            _, sample, _ = prior("prior", y_onehot_A, hps)
            z = sample(eps_std=eps_std)
            z = decoder_A(z, eps_std=eps_std)
            z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            x_A = postprocess(z)

        with tf.variable_scope('model_B', reuse=True):
            y_onehot_B = tf.cast(tf.one_hot(y_B, hps.n_y, 1, 0), 'float32')

            _, sample, _ = prior("prior", y_onehot_B, hps)
            z = sample(eps_std=eps_std)
            z = decoder_B(z, eps_std=eps_std)
            z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
            x_B = postprocess(z)
        return x_A, x_B

    m.eps_std = tf.placeholder(tf.float32, [None], name='eps_std')
    x_A_sampled, x_B_sampled = f_sample(Y_A, Y_B, m.eps_std)

    def sample_A(_y, _eps_std):
        return m.sess.run(x_A_sampled, {Y_A: _y, m.eps_std: _eps_std})
    def sample_B(_y, _eps_std):
        return m.sess.run(x_B_sampled, {Y_B: _y, m.eps_std: _eps_std})
    m.sample_A = sample_A
    m.sample_B = sample_B

    if hps.inference:
        # === Encoder-Decoder functions
        def f_encode(x, y, model_name, reuse=True):
            assert model_name == 'model_A' or model_name == 'model_B'
            encoder = encoder_A if model_name == 'model_A' else encoder_B
            with tf.variable_scope(model_name, reuse=reuse):
                y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')

                # Discrete -> Continuous
                objective = tf.zeros_like(x, dtype='float32')[:, 0, 0, 0]
                z = preprocess(x)
                z = z + tf.random_uniform(tf.shape(z), 0, 1. / hps.n_bins)
                objective += - np.log(hps.n_bins) * np.prod(Z.int_shape(z)[1:])

                # Encode
                z = Z.squeeze2d(z, 2)  # > 16x16x12
                z, objective, eps = encoder(z, objective)

                # Prior
                hps.top_shape = Z.int_shape(z)[1:]
                logp, _, _eps = prior("prior", y_onehot, hps)
                objective += logp(z)
                eps.append(_eps(z))

            return eps

        def f_decode(y, eps, model_name, reuse=True):
            assert model_name == 'model_A' or model_name == 'model_B'
            decoder = decoder_A if model_name == 'model_A' else decoder_B
            with tf.variable_scope(model_name, reuse=reuse):
                y_onehot = tf.cast(tf.one_hot(y, hps.n_y, 1, 0), 'float32')

                _, sample, _ = prior("prior", y_onehot, hps)
                z = sample(eps=eps[-1])
                z = decoder(z, eps=eps[:-1])
                z = Z.unsqueeze2d(z, 2)  # 8x8x12 -> 16x16x3
                x = postprocess(z)

            return x

        enc_eps_A, enc_eps_B = f_encode(X, Y, 'model_A'), f_encode(X, Y, 'model_B')
        dec_eps_A, dec_eps_B = [], []
        for enc_eps, dec_eps in zip([enc_eps_A, enc_eps_B], [dec_eps_A, dec_eps_B]):
            for i, _eps in enumerate(enc_eps):
                dec_eps.append(tf.placeholder(tf.float32, _eps.get_shape().as_list(), name="dec_eps_" + str(i)))
        dec_x_A = f_decode(Y, dec_eps_A, 'model_A')
        dec_x_B = f_decode(Y, dec_eps_B, 'model_B')

        eps_shapes = [_eps.get_shape().as_list()[1:] for _eps in enc_eps_A]

        def flatten_eps(eps):
            # [BS, eps_size]
            return np.concatenate([np.reshape(e, (e.shape[0], -1)) for e in eps], axis=-1)

        def unflatten_eps(feps):
            index = 0
            eps = []
            bs = feps.shape[0]
            for shape in eps_shapes:
                eps.append(np.reshape(feps[:, index: index+np.prod(shape)], (bs, *shape)))
                index += np.prod(shape)
            return eps

        # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)
        def encode(x, y, model_name):
            assert model_name == 'model_A' or model_name == 'model_B'
            if model_name == 'model_A':
                return flatten_eps(sess.run(enc_eps_A, {X_A: x, Y_A: y}))
            elif model_name == 'model_B':
                return flatten_eps(sess.run(enc_eps_B, {X_B: x, Y_B: y}))

        def decode(y, feps, model_name):
            assert model_name == 'model_A' or model_name == 'model_B'
            if model_name == 'model_A':
                eps_A = unflatten_eps(feps)
                feed_dict = {Y_A: y}
                for i in range(len(dec_eps_A)):
                    feed_dict[dec_eps_A[i]] = eps_A[i]
                return sess.run(dec_x_A, feed_dict)
            elif model_name == 'model_B':
                eps_B = unflatten_eps(feps)
                feed_dict = {Y_B: y}
                for i in range(len(dec_eps_B)):
                    feed_dict[dec_eps_B[i]] = eps_B[i]
                return sess.run(dec_x_B, feed_dict)

        m.encode = encode
        m.decode = decode

    return m


def checkpoint(z, logdet):
    zshape = Z.int_shape(z)
    z = tf.reshape(z, [-1, zshape[1]*zshape[2]*zshape[3]])
    logdet = tf.reshape(logdet, [-1, 1])
    combined = tf.concat([z, logdet], axis=1)
    tf.add_to_collection('checkpoints', combined)
    logdet = combined[:, -1]
    z = tf.reshape(combined[:, :-1], [-1, zshape[1], zshape[2], zshape[3]])
    return z, logdet


@add_arg_scope
def revnet2d(name, z, logdet, hps, reverse=False):
    with tf.variable_scope(name):
        if not reverse:
            for i in range(hps.depth):
                z, logdet = checkpoint(z, logdet)
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
            z, logdet = checkpoint(z, logdet)
        else:
            for i in reversed(range(hps.depth)):
                z, logdet = revnet2d_step(str(i), z, logdet, hps, reverse)
    return z, logdet

# Simpler, new version
@add_arg_scope
def revnet2d_step(name, z, logdet, hps, reverse):
    with tf.variable_scope(name):

        shape = Z.int_shape(z)
        n_z = shape[3]
        assert n_z % 2 == 0

        if not reverse:

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv("invconv", z, logdet)
            else:
                raise Exception()

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 += f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 += shift
                z2 *= scale
                logdet += tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

        else:

            z1 = z[:, :, :, :n_z // 2]
            z2 = z[:, :, :, n_z // 2:]

            if hps.flow_coupling == 0:
                z2 -= f("f1", z1, hps.width)
            elif hps.flow_coupling == 1:
                h = f("f1", z1, hps.width, n_z)
                shift = h[:, :, :, 0::2]
                # scale = tf.exp(h[:, :, :, 1::2])
                scale = tf.nn.sigmoid(h[:, :, :, 1::2] + 2.)
                z2 /= scale
                z2 -= shift
                logdet -= tf.reduce_sum(tf.log(scale), axis=[1, 2, 3])
            else:
                raise Exception()

            z = tf.concat([z1, z2], 3)

            if hps.flow_permutation == 0:
                z = Z.reverse_features("reverse", z, reverse=True)
            elif hps.flow_permutation == 1:
                z = Z.shuffle_features("shuffle", z, reverse=True)
            elif hps.flow_permutation == 2:
                z, logdet = invertible_1x1_conv(
                    "invconv", z, logdet, reverse=True)
            else:
                raise Exception()

            z, logdet = Z.actnorm("actnorm", z, logdet=logdet, reverse=True)

    return z, logdet


def f(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv2d("l_1", h, width))
        h = tf.nn.relu(Z.conv2d("l_2", h, width, filter_size=[1, 1]))
        h = Z.conv2d_zeros("l_last", h, n_out)
    return h


def f_resnet(name, h, width, n_out=None):
    n_out = n_out or int(h.get_shape()[3])
    with tf.variable_scope(name):
        h = tf.nn.relu(Z.conv2d("l_1", h, width))
        h = Z.conv2d_zeros("l_2", h, n_out)
    return h

# Invertible 1x1 conv
@add_arg_scope
def invertible_1x1_conv(name, z, logdet, reverse=False):

    if True:  # Set to "False" to use the LU-decomposed version

        with tf.variable_scope(name):

            shape = Z.int_shape(z)
            w_shape = [shape[3], shape[3]]

            # Sample a random orthogonal matrix:
            w_init = np.linalg.qr(np.random.randn(
                *w_shape))[0].astype('float32')

            w = tf.get_variable("W", dtype=tf.float32, initializer=w_init)

            # dlogdet = tf.linalg.LinearOperator(w).log_abs_determinant() * shape[1]*shape[2]
            dlogdet = tf.cast(tf.log(abs(tf.matrix_determinant(
                tf.cast(w, 'float64')))), 'float32') * shape[1]*shape[2]

            if not reverse:

                _w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += dlogdet

                return z, logdet
            else:

                _w = tf.matrix_inverse(w)
                _w = tf.reshape(_w, [1, 1]+w_shape)
                z = tf.nn.conv2d(z, _w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet -= dlogdet

                return z, logdet

    else:

        # LU-decomposed version
        shape = Z.int_shape(z)
        with tf.variable_scope(name):

            dtype = 'float64'

            # Random orthogonal matrix:
            import scipy
            np_w = scipy.linalg.qr(np.random.randn(shape[3], shape[3]))[
                0].astype('float32')

            np_p, np_l, np_u = scipy.linalg.lu(np_w)
            np_s = np.diag(np_u)
            np_sign_s = np.sign(np_s)
            np_log_s = np.log(abs(np_s))
            np_u = np.triu(np_u, k=1)

            p = tf.get_variable("P", initializer=np_p, trainable=False)
            l = tf.get_variable("L", initializer=np_l)
            sign_s = tf.get_variable(
                "sign_S", initializer=np_sign_s, trainable=False)
            log_s = tf.get_variable("log_S", initializer=np_log_s)
            # S = tf.get_variable("S", initializer=np_s)
            u = tf.get_variable("U", initializer=np_u)

            p = tf.cast(p, dtype)
            l = tf.cast(l, dtype)
            sign_s = tf.cast(sign_s, dtype)
            log_s = tf.cast(log_s, dtype)
            u = tf.cast(u, dtype)

            w_shape = [shape[3], shape[3]]

            l_mask = np.tril(np.ones(w_shape, dtype=dtype), -1)
            l = l * l_mask + tf.eye(*w_shape, dtype=dtype)
            u = u * np.transpose(l_mask) + tf.diag(sign_s * tf.exp(log_s))
            w = tf.matmul(p, tf.matmul(l, u))

            if True:
                u_inv = tf.matrix_inverse(u)
                l_inv = tf.matrix_inverse(l)
                p_inv = tf.matrix_inverse(p)
                w_inv = tf.matmul(u_inv, tf.matmul(l_inv, p_inv))
            else:
                w_inv = tf.matrix_inverse(w)

            w = tf.cast(w, tf.float32)
            w_inv = tf.cast(w_inv, tf.float32)
            log_s = tf.cast(log_s, tf.float32)

            if not reverse:

                w = tf.reshape(w, [1, 1] + w_shape)
                z = tf.nn.conv2d(z, w, [1, 1, 1, 1],
                                 'SAME', data_format='NHWC')
                logdet += tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet
            else:

                w_inv = tf.reshape(w_inv, [1, 1]+w_shape)
                z = tf.nn.conv2d(
                    z, w_inv, [1, 1, 1, 1], 'SAME', data_format='NHWC')
                logdet -= tf.reduce_sum(log_s) * (shape[1]*shape[2])

                return z, logdet


@add_arg_scope
def split2d(name, z, objective=0.):
    with tf.variable_scope(name):
        n_z = Z.int_shape(z)[3]
        z1 = z[:, :, :, :n_z // 2]
        z2 = z[:, :, :, n_z // 2:]
        pz = split2d_prior(z1)
        objective += pz.logp(z2)
        z1 = Z.squeeze2d(z1)
        eps = pz.get_eps(z2)
        return z1, objective, eps


@add_arg_scope
def split2d_reverse(name, z, eps, eps_std):
    with tf.variable_scope(name):
        z1 = Z.unsqueeze2d(z)
        pz = split2d_prior(z1)
        if eps is not None:
            # Already sampled eps
            z2 = pz.sample2(eps)
        elif eps_std is not None:
            # Sample with given eps_std
            z2 = pz.sample2(pz.eps * tf.reshape(eps_std, [-1, 1, 1, 1]))
        else:
            # Sample normally
            z2 = pz.sample
        z = tf.concat([z1, z2], 3)
        return z


@add_arg_scope
def split2d_prior(z):
    n_z2 = int(z.get_shape()[3])
    n_z1 = n_z2
    h = Z.conv2d_zeros("conv", z, 2 * n_z1)

    mean = h[:, :, :, 0::2]
    logs = h[:, :, :, 1::2]
    return Z.gaussian_diag(mean, logs)
