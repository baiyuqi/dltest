#-*-coding:utf-8-*-
from PIL import Image,ImageFilter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
img=Image.open("/1ee25-1545795530552.jpg")
im=img.convert("L")

a = np.array(im)
pd.DataFrame(a.sum(axis=0)).plot.line() # 画出每列的像素累计值
plt.imshow(a,cmap='gray') # 画出图像


# 核心代码，注意调整要切割的线
split_lines = [5,16,35,48,66]
vlines = [plt.axvline(i, color='r') for i in split_lines] # 画出分割线
plt.show()

'''
#################核心代码##########################
'''

#设置获取图像的高和宽,根据需要调整

y_min=1
y_max=23

ims=[]
c=1
for x_min,x_max in zip(split_lines[:-1],split_lines[1:]):
   im.crop([x_min,y_min,x_max,y_max] ).save(str(c)+'.jpeg')
   # crop()函数是截取指定图像！
   # save保存图像！
   c=c+1
for i in range(1,5):
   file_name="{}.jpeg".format(i)
   plt.subplot(8,3,i)
   im=Image.open(file_name).convert("1")
   #im=img.filter(ImageFilter.MedianFilter(size=3))
   plt.imshow(im)
   # 显示截取的图像！
plt.show()