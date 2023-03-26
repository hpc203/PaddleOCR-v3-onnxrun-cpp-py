import onnxruntime
import numpy as np
import cv2
import math

class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + ' '  # for `-1` index
        self.dict = {}
        for i, char in enumerate(alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[char] = i + 1
    def decode(self, t, length, raw=False):
        t = t[:length]
        if raw:
            return ''.join([self.alphabet[i - 1] for i in t])
        else:
            char_list = []
            for i in range(length):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):
                    char_list.append(self.alphabet[t[i] - 1])
            return ''.join(char_list)

class TextRecognizer:
    def __init__(self):
        self.sess = onnxruntime.InferenceSession('weights/ch_PP-OCRv3_rec_infer.onnx')
        self.alphabet = list(map(lambda x:x.decode('utf-8').strip("\n").strip("\r\n"), open('rec_word_dict.txt', 'rb').readlines()))
        self.converter = strLabelConverter(''.join(self.alphabet))
        self.rec_image_shape = [3, 48, 320]

    def resize_norm_img(self, img):
        imgC, imgH, imgW = self.rec_image_shape
        # max_wh_ratio = imgW / imgH
        # assert imgC == img.shape[2]
        # imgW = int((imgH * max_wh_ratio))

        h, w = img.shape[:2]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))

        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    def predict_text(self, im):
        img = self.resize_norm_img(im)
        transformed_image = np.expand_dims(img, axis=0)

        ort_inputs = {i.name: transformed_image for i in self.sess.get_inputs()}
        preds = self.sess.run(None, ort_inputs)
        preds = preds[0].squeeze(axis=0)
        length  = preds.shape[0]
        preds = preds.reshape(length,-1)
        # preds = softmax(preds)
        preds = np.argmax(preds,axis=1)
        preds = preds.reshape(-1)
        sim_pred = self.converter.decode(preds, length, raw=False)
        return sim_pred