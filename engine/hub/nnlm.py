import tensorflow as tf
import tensorflow_hub as hub
import hashlib
import os
#handle = "https://tfhub.dev/google/imagenet/inception_v1/feature_vector/1"
#hashlib.sha1(handle.encode("utf8")).hexdigest()
#os.environ["TFHUB_CACHE_DIR"] = 'F:\hubmodules'

with tf.Graph().as_default():
  #module_url = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
  embed = hub.Module("d:/downloaded/nnl")
  embeddings = embed(["A long sentence.", "single-word",
                      "http://example.com"])

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    print(sess.run(embeddings))