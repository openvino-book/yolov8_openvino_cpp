#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <openvino/openvino.hpp> //openvino header file
#include <opencv2/opencv.hpp>    //opencv header file

using namespace cv;
using namespace dnn;

std::vector<Scalar> colors = { Scalar(255, 0, 0), Scalar(255, 0, 255), Scalar(170, 0, 255), Scalar(255, 0, 85),
                                   Scalar(255, 0, 170), Scalar(85, 255, 0), Scalar(255, 170, 0), Scalar(0, 255, 0),
                                   Scalar(255, 255, 0), Scalar(0, 255, 85), Scalar(170, 255, 0), Scalar(0, 85, 255),
                                   Scalar(0, 255, 170), Scalar(0, 0, 255), Scalar(0, 255, 255), Scalar(85, 0, 255)};

const std::vector<std::string> class_names = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush" };

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

float sigmoid_function(float a){
    float b = 1. / (1. + exp(-a));
    return b;
}

int main(int argc, char* argv[])
{
    // -------- Step 1. Initialize OpenVINO Runtime Core --------
    ov::Core core;

    // -------- Step 2. Compile the Model --------
    auto compiled_model = core.compile_model("yolov8n-seg.xml", "CPU");

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
    auto output0 = infer_request.get_output_tensor(0); //output0
    auto output1 = infer_request.get_output_tensor(1); //otuput1
    auto output0_shape = output0.get_shape();
    auto output1_shape = output1.get_shape();
    std::cout << "The shape of output0:" << output0_shape << std::endl;
    std::cout << "The shape of output1:" << output1_shape << std::endl;

    // -------- Step 8. Postprocess the result --------
    Mat output_buffer(output0_shape[1], output0_shape[2], CV_32F, output0.data<float>());
    Mat proto(32, 25600, CV_32F, output1.data<float>()); //[32,25600]
    transpose(output_buffer, output_buffer); //[8400,116]
    float score_threshold = 0.25;
    float nms_threshold = 0.5;
    std::vector<int> class_ids;
    std::vector<float> class_scores;
    std::vector<Rect> boxes;
    std::vector<Mat> mask_confs;
    // Figure out the bbox, class_id and class_score
    for (int i = 0; i < output_buffer.rows; i++) {
        Mat classes_scores = output_buffer.row(i).colRange(4, 84);
        Point class_id;
        double maxClassScore;
        minMaxLoc(classes_scores, 0, &maxClassScore, 0, &class_id);

        if (maxClassScore > score_threshold) {
            class_scores.push_back(maxClassScore);
            class_ids.push_back(class_id.x);
            float cx = output_buffer.at<float>(i, 0);
            float cy = output_buffer.at<float>(i, 1);
            float w = output_buffer.at<float>(i, 2);
            float h = output_buffer.at<float>(i, 3);

            int left = int((cx - 0.5 * w) * scale);
            int top = int((cy - 0.5 * h) * scale);
            int width = int(w * scale);
            int height = int(h * scale);

            cv::Mat mask_conf = output_buffer.row(i).colRange(84, 116);
            mask_confs.push_back(mask_conf);
            boxes.push_back(Rect(left, top, width, height));
        }
    }
    //NMS
    std::vector<int> indices;
    NMSBoxes(boxes, class_scores, score_threshold, nms_threshold, indices);

    // -------- Visualize the detection results -----------
    cv::Mat rgb_mask = cv::Mat::zeros(img.size(), img.type());
    cv::Mat masked_img;
    cv::RNG rng;

    for (size_t i = 0; i < indices.size(); i++) {
        // Visualize the objects
        int index = indices[i];
        int class_id = class_ids[index];
        rectangle(img, boxes[index], colors[class_id % 16], 2, 8);
        std::string label = class_names[class_id] + ":" + std::to_string(class_scores[index]).substr(0, 4);
        Size textSize = cv::getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, 0);
        Rect textBox(boxes[index].tl().x, boxes[index].tl().y - 15, textSize.width, textSize.height+5);
        cv::rectangle(img, textBox, colors[class_id % 16], FILLED);
        putText(img, label, Point(boxes[index].tl().x, boxes[index].tl().y - 5), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

        // Visualize the Masks
        Mat m = mask_confs[i] * proto;
        for (int col = 0; col < m.cols; col++) {
            m.at<float>(0, col) = sigmoid_function(m.at<float>(0, col));
        }
        cv::Mat m1 = m.reshape(1, 160); // 1x25600 -> 160x160
        int x1 = std::max(0, boxes[index].x);
        int y1 = std::max(0, boxes[index].y);
        int x2 = std::max(0, boxes[index].br().x);
        int y2 = std::max(0, boxes[index].br().y);
        int mx1 = int(x1 / scale * 0.25);
        int my1 = int(y1 / scale * 0.25);
        int mx2 = int(x2 / scale * 0.25);
        int my2 = int(y2 / scale * 0.25);

        cv::Mat mask_roi = m1(cv::Range(my1, my2), cv::Range(mx1, mx2));
        cv::Mat rm, det_mask;
        cv::resize(mask_roi, rm, cv::Size(x2 - x1, y2 - y1));

        for (int r = 0; r < rm.rows; r++) {
            for (int c = 0; c < rm.cols; c++) {
                float pv = rm.at<float>(r, c);
                if (pv > 0.5) {
                    rm.at<float>(r, c) = 1.0;
                }
                else {
                    rm.at<float>(r, c) = 0.0;
                }
            }
        }
        rm = rm * rng.uniform(0, 255);
        rm.convertTo(det_mask, CV_8UC1);
        if ((y1 + det_mask.rows) >= img.rows) {
            y2 = img.rows - 1;
        }
        if ((x1 + det_mask.cols) >= img.cols) {
            x2 = img.cols - 1;
        }

        cv::Mat mask = cv::Mat::zeros(cv::Size(img.cols, img.rows), CV_8UC1);
        det_mask(cv::Range(0, y2 - y1), cv::Range(0, x2 - x1)).copyTo(mask(cv::Range(y1, y2), cv::Range(x1, x2)));
        add(rgb_mask, cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255)), rgb_mask, mask);
        addWeighted(img, 0.5, rgb_mask, 0.5, 0, masked_img);
    }

    namedWindow("YOLOv8-Seg OpenVINO Inference C++ Demo", WINDOW_AUTOSIZE);
    imshow("YOLOv8-Seg OpenVINO Inference C++ Demo", masked_img);
    waitKey(0);
    destroyAllWindows();
    return 0;
}