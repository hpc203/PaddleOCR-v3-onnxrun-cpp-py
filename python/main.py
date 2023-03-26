import os
import argparse
import cv2
import numpy as np
from text_det import TextDetector
from text_angle_cls import TextClassifier
from text_rec import TextRecognizer
os.environ["KMP_DUPLICATE_LIB_OK"] = 'TRUE'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--imgpath', type=str, default='images/1.jpg', help="image path")
    args = parser.parse_args()

    detect_model = TextDetector()
    angle_model = TextClassifier()
    rec_model = TextRecognizer()

    srcimg = cv2.imread(args.imgpath)
    # srcimg = cv2.rotate(srcimg, 1)
    box_list = detect_model.detect(srcimg)
    text = ''
    if len(box_list) > 0:
        for point in box_list:
            point = detect_model.order_points_clockwise(point)
            textimg = detect_model.get_rotate_crop_image(srcimg, point.astype(np.float32))
            angle = angle_model.predict(textimg)
            if angle=='180':
                textimg = cv2.rotate(textimg, 1)
            text = rec_model.predict_text(textimg)

            point = point.astype(int)
            cv2.polylines(srcimg, [point], True, (0, 0, 255), thickness=2)
            for i in range(4):
                cv2.circle(srcimg, tuple(point[i, :]), 3, (0, 255, 0), thickness=-1)
            print(text)

    winName = 'Deep learning object detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.imwrite('result.jpg', srcimg)