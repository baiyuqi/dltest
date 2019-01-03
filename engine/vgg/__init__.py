from matplotlib import pyplot as plt
import numpy as np
import os
import tensorflow as tf
import urllib.error
import urllib.request as req
import tensorflow.contrib.slim as slim
import sys
sys.path.append("/repository/models/research/slim")
from datasets import imagenet
from nets import vgg
from preprocessing import vgg_preprocessing
import io, json
label_path="/tensorflow/imagenetlabels/labels.json"
# names = imagenet.create_readable_names_for_imagenet_labels()
# with io.open('/tensorflow/imagenetlabels/labels.json', 'w', encoding='utf-8') as f:
#   f.write(json.dumps(names, ensure_ascii=False))
json1_file = open('/repository/labels.json')
json1_str = json1_file.read()
names = json.loads(json1_str)