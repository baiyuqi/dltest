import cv2
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
def binarize_image_using_opencv(captcha_path, binary_image_path='input-black-n-white.jpg'):
     im_gray = cv2.imread(captcha_path, cv2.IMREAD_GRAYSCALE)
     (thresh, im_bw) = cv2.threshold(im_gray, 85, 255, cv2.THRESH_BINARY)
     # although thresh is used below, gonna pick something suitable
     im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
     cv2.imwrite(binary_image_path, im_bw)

     return binary_image_path

def preprocess_image_using_opencv(captcha_path):
     bin_image_path = binarize_image_using_opencv(captcha_path)

     im_bin = Image.open(bin_image_path)
     basewidth = 300  # in pixels
     wpercent = (basewidth/float(im_bin.size[0]))
     hsize = int((float(im_bin.size[1])*float(wpercent)))
     big = im_bin.resize((basewidth, hsize), Image.NEAREST)

     # tesseract-ocr only works with TIF so save the bigger image in that format
     tif_file = "input-NEAREST.tif"
     big.save(tif_file)

     return tif_file

def get_captcha_text_from_captcha_image(captcha_path):

     # Preprocess the image befor OCR
     tif_file = preprocess_image_using_opencv(captcha_path)



get_captcha_text_from_captcha_image("path/captcha.png")

im = Image.open("input-NEAREST.tif") # the second one
im = im.filter(ImageFilter.MedianFilter())
enhancer = ImageEnhance.Contrast(im)
im = enhancer.enhance(2)
im = im.convert('1')
im.save('captchafinal.tif')
text = pytesseract.image_to_string(Image.open('captchafinal.tif'), config="-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ -psm 6")
print(text)
