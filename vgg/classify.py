from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib.error
import urllib.request as req
import sys

from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing
from classifier.vgg import names



checkpoints_dir = '/repository/checkpoints'
image_size = vgg.vgg_16.default_image_size
slim = tf.contrib.slim


def classify(url):

    with tf.Graph().as_default():
        image_string = req.urlopen(url).read()
        image = tf.image.decode_jpeg(image_string, channels=3)
        processed_image = vgg_preprocessing.preprocess_image(image,
                                                             image_size,
                                                             image_size,
                                                             is_training=False)
        processed_images = tf.expand_dims(processed_image, 0)

        # Create the model, use the default arg scope to configure
        # the batch norm parameters. arg_scope is a very conveniet
        # feature of slim library -- you can define default
        # parameters for layers -- like stride, padding etc.
        with slim.arg_scope(vgg.vgg_arg_scope()):
            logits, _ = vgg.vgg_16(processed_images,
                                   num_classes=1000,
                                   is_training=False)

        # In order to get probabilities we apply softmax on the output.
        probabilities = tf.nn.softmax(logits)

        # Create a function that reads the network weights
        # from the checkpoint file that you downloaded.
        # We will run it in session later.
        init_fn = slim.assign_from_checkpoint_fn(
            os.path.join(checkpoints_dir, 'vgg_16.ckpt'),
            slim.get_model_variables('vgg_16'))

        with tf.Session() as sess:
            writer = tf.summary.FileWriter("/temp/logs", sess.graph)
            # Load weights
            init_fn(sess)

            # We want to get predictions, image as numpy matrix
            # and resized and cropped piece that is actually
            # being fed to the network.
            np_image, network_input, probabilities = sess.run([image,
                                                               processed_image,
                                                               probabilities])
            probabilities = probabilities[0, 0:]
            sorted_inds = [i[0] for i in sorted(enumerate(-probabilities),
                                                key=lambda x: x[1])]
        rst = ""
        for i in range(5):
            index = sorted_inds[i]
            pos = probabilities[index]
            name = names[str(index + 1)]
            rst += (name + ":" + str(pos) + "\n")
        return rst
