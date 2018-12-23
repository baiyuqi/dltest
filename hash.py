import hashlib

handle = "https://tfhub.dev/google/nnlm-en-dim128-with-normalization/1"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())
handle = "https://tfhub.dev/deepmind/biggan-512/2"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())
handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/classification/2"
print(hashlib.sha1(handle.encode("utf8")).hexdigest())
