# PaddleOCR-v3-onnxrun-cpp-py
使用ONNXRuntime部署PaddleOCR-v3, 包含C++和Python两个版本的程序。

从百度PaddlePaddle团队的PaddleOCR项目里导出的onnx文件，使用onnxruntime部署，
从而摆脱对深度学习框架PaddlePaddle的依赖。起初想用opnecv部署的，可是opencv的
dnn模块读取onnx文件出错了，无赖只能使用onnxruntime部署。
本套程序里包含dbnet文件检测，文字方向分类，crnn文字识别三个模块，
onnx文件大小不超过15M
