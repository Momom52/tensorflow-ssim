import numpy as np
import cv2
import tensorflow as tf
from PIL import Image

def img_resize(img1, img2, width=641, height=641):
    """
    The function is to uniform the img1 and the img2's shape
    """
    img1 = Image.fromarray(img1)
    img2 = Image.fromarray(img2)
    new_img1 = np.asarray(img1.resize((width, height), Image.BILINEAR))
    new_img2 = np.asarray(img2.resize((width, height), Image.BILINEAR))
    #new_img1.show()
    return new_img1, new_img2

def gaussian(window_size, sigma = 1.5):
    window = [np.exp(-(x - window_size//2)**2 / float(2*sigma**2)) for x in range(window_size)]
    window = [window, window]
    #print(window)
    window = tf.convert_to_tensor(window)
    window /= tf.reduce_sum(window, 0)
    window = tf.expand_dims(window, -1)
    window = tf.expand_dims(window, -1)
    #print(window)

    return window

def ssim(img1, img2, k1=0.01, k2=0.02, L=255, window_size=11):
    """
    The function is to calculate the ssim score
    Img1's shape should be equal to img2's shape

    """
    if not img1.shape == img2.shape:
        img1, img2 = img_resize(img1, img2)

    if img1.dtype == np.uint8:
        img1 = np.float32(img1)
    if img2.dtype == np.uint8:
        img2 = np.float32(img2)

    h,w = img1.shape

    img1 = tf.expand_dims(img1, 0)
    img1 = tf.expand_dims(img1, -1)
    img2 = tf.expand_dims(img2, 0)
    img2 = tf.expand_dims(img2, -1)

    window = gaussian(window_size)

    mu1 = tf.nn.conv2d(img1, window, strides = [1, h, w, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, h, w, 1], padding = 'VALID')

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,h, w, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, h, w, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, h, w, 1], padding = 'VALID') - mu1_mu2

    c1 = (k1*L)**2
    c2 = (k2*L)**2

    with tf.Session() as sess:
        sess.run(mu1)
        sess.run(mu2)
        sess.run(sigma1_sq)
        sess.run(sigma2_sq)
        sess.run(sigma1_2)
        ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
        ssim_map = sess.run(ssim_map)

    return np.mean(np.mean(ssim_map))


if __name__ == '__main__':
    img1 = np.asarray(cv2.imread(path, 0))
    img2 = np.asarray(cv2.imread(path, 0))
    score = ssim(img1,img2)

    print(score)
