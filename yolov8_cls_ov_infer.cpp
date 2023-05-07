#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

using namespace cv;
using namespace dnn;

// Keep the ratio before resize
Mat letterbox(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    Mat result = Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(Rect(0, 0, col, row)));
    return result;
}

int main(int argc, char* argv[])
{
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    auto compiled_model = core.compile_model("yolov8n-cls.xml", "CPU");

    // -------- Step 3. Create an Inference Request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Step 4.Read a picture file and do the preprocess --------
    Mat img = cv::imread("bus.jpg"); 
    // Preprocess the image
    Mat letterbox_img = letterbox(img);
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(224, 224), Scalar(), true);

    // -------- Step 5. Feed the blob into the input node of the Model -------
    // Get input port for model with one input
    auto input_port = compiled_model.input();
    // Create tensor from external memory
    ov::Tensor input_tensor(input_port.get_element_type(), input_port.get_shape(), blob.ptr(0));
    // Set input tensor for model with one input
    infer_request.set_input_tensor(input_tensor);

    // -------- Step 6. Start inference --------
    infer_request.infer();

    // -------- Step 7. Get the inference result --------
    auto output = infer_request.get_output_tensor(0);
    auto output_shape = output.get_shape();
    std::cout << "The shape of output tensor:" << output_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    float* output_buffer = output.data<float>();
    std::vector<float> result(output_buffer, output_buffer + output_shape[1]);
    auto max_idx = std::max_element(result.begin(), result.end());
    int class_id = max_idx - result.begin();
    float score = *max_idx;
    std::cout << "Class ID:" << class_id << " Score:" <<score<< std::endl;
    
    return 0;
}
