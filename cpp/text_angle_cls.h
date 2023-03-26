#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <fstream>
#include <numeric>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
//#include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;
using namespace Ort;

class TextClassifier
{
public:
	TextClassifier();
	int predict(Mat cv_image);
private:
	const int label_list[2] = { 0, 180 };

	Mat preprocess(Mat srcimg);
	void normalize_(Mat img);
	const int inpWidth = 192;
	const int inpHeight = 48;
	int num_out;
	vector<float> input_image_;

	Env env = Env(ORT_LOGGING_LEVEL_ERROR, "Angle classify");
	Ort::Session *ort_session = nullptr;
	SessionOptions sessionOptions = SessionOptions();
	vector<char*> input_names;
	vector<char*> output_names;
	vector<vector<int64_t>> input_node_dims; // >=1 outputs
	vector<vector<int64_t>> output_node_dims; // >=1 outputs
};