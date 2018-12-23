
import tensorflow as tf
import tensorflow_hub as hub
import hashlib
import os
import json
import scipy
#handle = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1"
#hashlib.sha1(handle.encode("utf8")).hexdigest()
os.environ["TFHUB_CACHE_DIR"] = 'F:\hubmodules'
json1_file = open('./data/labels.json')
json1_str = json1_file.read()
names = json.loads(json1_str)
import numpy
with tf.Graph().as_default():
    module = hub.Module("https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2")
    height, width = hub.get_expected_image_size(module)
    images = tf.placeholder(dtype = numpy.float32, shape=(1, height, width, 3))

    data = scipy.misc.imread('./images/3-1.jpg')
    data = scipy.resize(data, (1, height, width, 3))
    # A batch of images with shape [batch_size, height, width, 3].
    logits = module(images)  # Logits with shape [batch_size, num_classes].
    probabilities = tf.nn.softmax(logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        prob = sess.run(probabilities, feed_dict={images:data})
        prob = prob[0, 0:]
        print(prob);
        sorted_inds = [i[0] for i in sorted(enumerate(-prob),
                                            key=lambda x: x[1])]
        rst = ""
        for i in range(5):
            index = sorted_inds[i]
            pos = prob[index]
            name = names[str(index + 1)]
            rst += (name + ":" + str(pos) + "\n")
        print(rst)