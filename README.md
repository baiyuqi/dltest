# dltest
deep learning test

迁移学习的实现技术问题
1. 首先，训练的是bottleneck之后的全连接附加层，其他层被冻结。所以bottleneck是不变的。
2. bottleneck与附加模型之间通过tf.placeholder_with_default相连接，用于训练和eval。
总之这是非常浅层的迁移训练

torch:
CUDA 9.0

To install PyTorch via pip, and you are using CUDA 9.0, use the following command, depending on your Python version:

# Python 3.5
pip3 install https://download.pytorch.org/whl/cu90/torch-1.0.0-cp35-cp35m-win_amd64.whl
pip3 install torchvision