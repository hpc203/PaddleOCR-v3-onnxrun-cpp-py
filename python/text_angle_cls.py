import onnxruntime
import math
import cv2
import numpy as np

class TextClassifier:
    def __init__(self):
        self.sess = onnxruntime.InferenceSession('weights/ch_ppocr_mobile_v2.0_cls_train.onnx')
        self.cls_image_shape = [3, 48, 192]
        self.label_list = ['0', '180']

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.cls_image_shape

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if imgC == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def predict(self, im):
        img = self.resize_norm_img(im)
        transformed_image = np.expand_dims(img, axis=0)

        ort_inputs = {i.name: transformed_image for i in self.sess.get_inputs()}
        preds = self.sess.run(None, ort_inputs)
        preds = preds[0].squeeze(axis=0)
        pred_idxs = preds.argmax()
        return self.label_list[pred_idxs]