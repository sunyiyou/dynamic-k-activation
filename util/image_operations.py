import numpy as np
from scipy.misc import imresize, imread, imsave
import cv2
import PIL

size_upsample = (256, 256)


def imconcat(imgs, margin=0):
    w = sum([img.width for img in imgs]) + (len(imgs) - 1) * margin
    ret = PIL.Image.new("RGB", (w, imgs[0].height), (255,255,255))
    w_pre = 0
    for i, img in enumerate(imgs):
        ret.paste(img, (w_pre+margin*i, 0))
        w_pre += img.width
    return ret

def imstack(imgs):
    h = sum([img.height for img in imgs])
    ret = PIL.Image.new("RGB", (imgs[0].width, h), (255,255,255))
    h_pre = 0
    for i, img in enumerate(imgs):
        ret.paste(img, (0, h_pre))
        h_pre += img.height
    return ret


def imagalize(mat):
    mat = mat - np.min(mat)
    mat_img = mat / np.max(mat)
    mat_img = np.uint8(255 * mat_img)
    mat_img = imresize(mat_img, size_upsample)
    return mat_img


def CAM(feature_conv, weight, class_idx):
    # generate the class activation maps upsample to 256x256
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for idx in class_idx:
        cam = weight[[idx]].mm(feature_conv.view((nc, h*w)))
        cam = cam.reshape(h, w).numpy()
        output_cam.append(imagalize(cam))
    return output_cam
