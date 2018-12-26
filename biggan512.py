
import tensorflow as tf
import tensorflow_hub as hub
import hashlib
import os
import time
import scipy



sess = None;
samples = None
with tf.Graph().as_default():
    module = hub.Module('d:/downloaded/bigan')
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    # Sample random noise (z) and ImageNet label (y) inputs.
    batch_size = 1
    truncation = 0.5  # scalar truncation value in [0.02, 1.0]
    z = truncation * tf.random.truncated_normal([batch_size, 128])  # noise sample
    y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
    y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

    samples = module(dict(y=y, z=z, truncation=truncation))

def create():
    images = sess.run(samples)
    t = time.time()
    fname = 'd:/images/' + str(t) + '.jpg'
    scipy.misc.imsave(fname, images[0, :, :, :])
    return  fname