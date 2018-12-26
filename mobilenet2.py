
import tensorflow as tf
import tensorflow_hub as hub
import hashlib
import os
import json
import numpy
import scipy

json1_file = open('./labels.json')
json1_str = json1_file.read()
names = json.loads(json1_str)
#handle = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1"
#hashlib.sha1(handle.encode("utf8")).hexdigest()
#os.environ["TFHUB_CACHE_DIR"] = 'F:\hubmodules'
def handle(image_data,resize_tensor,image_tensor):

    resized_input_values = sess.run(resize_tensor,
                                    {image_tensor: image_data})
    return resized_input_values

def preprocess(input_width, input_height):
    jpeg_data = tf.placeholder(tf.string, name='DecodeJPGInput')
    decoded_image = tf.image.decode_jpeg(jpeg_data, channels=3)
    # Convert from full range of uint8 to range [0,1] of float32.
    decoded_image_as_float = tf.image.convert_image_dtype(decoded_image,
                                                          tf.float32)
    decoded_image_4d = tf.expand_dims(decoded_image_as_float, 0)
    resize_shape = tf.stack([input_height, input_width])
    resize_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(decoded_image_4d,
                                             resize_shape_as_int)
    return jpeg_data, resized_image
def predict(image_data):
    resized = handle(image_data, resized_image, jpeg_data)
    prob = sess.run(fetches=probabilities, feed_dict={resized_image:resized})
    prob = prob[0, 0:]
    sorted_inds = [i[0] for i in sorted(enumerate(-prob),
                                        key=lambda x: x[1])]


    print(sorted_inds)

    rst = ""
    for i in range(2):
        index = sorted_inds[i]
        pos = prob[index]
        name = names[str(index+1)]
        rst += (name + ":" + str(pos) + "\n")
    print(rst)
    return rst
jpeg_data=None
resized_image=None
probabilities = None
sess = None
with tf.Graph().as_default():
    module = hub.Module("d:/downloaded/mobilenet")
    height, width = hub.get_expected_image_size(module)
    jpeg_data, resized_image = preprocess(width, height)


    logits = module(resized_image)  # Logits with shape [batch_size, num_classes].
    probabilities = tf.nn.softmax(logits)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())



#image_data = tf.gfile.FastGFile("d:/images/0.jpg", 'rb').read()
#predict(image_data)
