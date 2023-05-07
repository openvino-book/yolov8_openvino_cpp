#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

using namespace cv;
using namespace dnn;

//Colors for 17 keypoints
std::vector<cv::Scalar> colors = { Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
                                   Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
                                   Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
                                   Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255),
                                   Scalar(0, 170, 255)};

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
    auto compiled_model = core.compile_model("yolov8n-pose.xml", "CPU");

    // -------- Step 3. Create an Inference Request --------
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // -------- Step 4.Read a picture file and do the preprocess --------
    Mat img = cv::imread("bus.jpg");
    // Preprocess the image
    Mat letterbox_img = letterbox(img);
    float scale = letterbox_img.size[0] / 640.0;
    Mat blob = blobFromImage(letterbox_img, 1.0 / 255.0, Size(640, 640), Scalar(), true);

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
    float* data = output.data<float>();
    Mat output_buffer(output_shape[1], output_shape[2], CV_32F, data);
    transpose(output_buffer, output_buffer); //[8400,56]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<std::vector<float>> objects_keypoints;

    // //56: box[cx, cy, w, h] + Score + [17,3] keypoints
    for (int i = 0; i < output_buffer.rows; i++) {
        float class_score = output_buffer.at<float>(i, 4);

        if (class_score > score_threshold) {
            class_scores.push_back(class_score);
            class_ids.push_back(0); //{0:"person"}
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);
            // Get the box
            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);
            // Get the keypoints
            std::vector<float> keypoints;
            Mat kpts = output_buffer.row(i).colRange(5, 56);
            for (int i = 0; i < 17; i++) {                
                float x = kpts.at<float>(0, i * 3 + 0) * scale;
                float y = kpts.at<float>(0, i * 3 + 1) * scale;
                float s = kpts.at<float>(0, i * 3 + 2);
                keypoints.push_back(x);
                keypoints.push_back(y);
                keypoints.push_back(s);
            }

            boxes.push_back(Rect(left, top, width, height));
            objects_keypoints.push_back(keypoints);
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    // -------- Visualize the detection results -----------
    for (size_t i = 0; i < indices.size(); i++) {
        int index = indices[i];
        // Draw bounding box
        rectangle(img, boxes[index], Scalar(0, 0, 255), 2, 8);
        std::string label = "Person:" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height+5);
        cv::rectangle(img, textBox, Scalar(0, 0, 255), FILLED);
        putText(img, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));
        // Draw keypoints
        std::vector<float> object_keypoints = objects_keypoints[index];
        for (int i = 0; i < 17; i++) {
            int x = std::clamp(int(object_keypoints[i*3+0]), 0, img.cols);
            int y = std::clamp(int(object_keypoints[i*3+1]), 0, img.rows);
            //Draw point
            circle(img, Point(x, y), 5, colors[i], -1);
        }
    }
    namedWindow("YOLOv8-Pose OpenVINO Inference C++ Demo", WINDOW_AUTOSIZE);
    imshow("YOLOv8-Pose OpenVINO Inference C++ Demo", img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}