
import tensorflow as tf
import tensorflow_hub as hub
import hashlib
import os
import scipy
#handle = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1"
#hashlib.sha1(handle.encode("utf8")).hexdigest()
#os.environ["TFHUB_CACHE_DIR"] = 'F:\hubmodules'

with tf.Graph().as_default():
    # Load BigGAN 512 module.
    #module = hub.Module('https://tfhub.dev/deepmind/biggan-512/2')
    module = hub.Module('d:/downloaded/biggan')

    # Sample random noise (z) and ImageNet label (y) inputs.
    batch_size = 2
    truncation = 0.5  # scalar truncation value in [0.02, 1.0]
    z = truncation * tf.random.truncated_normal([batch_size, 128])  # noise sample
    y_index = tf.random.uniform([batch_size], maxval=1000, dtype=tf.int32)
    y = tf.one_hot(y_index, 1000)  # one-hot ImageNet label

    # Call BigGAN on a dict of the inputs to generate a batch of images with shape
    # [8, 512, 512, 3] and range [-1, 1].
    samples = module(dict(y=y, z=z, truncation=truncation))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        for k in range(10):
            images = sess.run(samples)
            for i in range(len(images)):
                scipy.misc.imsave('./images/' + str(k) + "-" + str(i) + '.jpg', images[i, :, :, :])
