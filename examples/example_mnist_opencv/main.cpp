#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include "inference/nn.h"

int main() {
    nnInit();
    NNModel* model;
    if (int rc = nnLoad("model-mnist.bin", &model, 1); rc != NN_OK) {
        std::cerr << "Model load failed: " << rc << "\n";
        return -1;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open video source.\n";
        return -1;
    }

    cv::Mat frame;
    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(gray, gray, { 5, 5 }, 0);

        cv::Mat thresh;
        cv::adaptiveThreshold(gray, thresh, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 11, 2);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, { 3, 3 });
        cv::morphologyEx(thresh, thresh, cv::MORPH_OPEN, kernel);

        cv::imshow("thresh", thresh);

        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(thresh, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (std::size_t i = 0; i < contours.size(); i++) {
            cv::Rect rect = cv::boundingRect(contours[i]);
            int x = rect.x, y = rect.y, w = rect.width, h = rect.height;
            if (w <= 0 || h <= 0) {
                continue;
            }

            if (w < 20 && h < 20) {
                continue;
            }
            if (w > frame.cols * 0.5 || h > frame.rows * 0.5) {
                continue;
            }

            cv::Mat digit = thresh(cv::Rect(x, y, w, h));

            // Resize to 20x20 preserving aspect ratio
            float scale = 20.0f / std::max(w, h);

            int scaledW = std::clamp(static_cast<int>(std::round(w * scale)), 1, 20);
            int scaledH = std::clamp(static_cast<int>(std::round(h * scale)), 1, 20);
            cv::Mat resized;
            cv::resize(digit, resized, cv::Size(scaledW, scaledH));

            // Pad to exactly 20x20, centering the digit
            int padTop = (20 - scaledH) / 2;
            int padBottom = 20 - scaledH - padTop;
            int padLeft = (20 - scaledW) / 2;
            int padRight = 20 - scaledW - padLeft;
            cv::Mat square20;
            cv::copyMakeBorder(resized, square20, padTop, padBottom, padLeft, padRight, cv::BORDER_CONSTANT, 0);

            // Add 4px border on each side to reach 28x28
            cv::Mat square28;
            cv::copyMakeBorder(square20, square28, 4, 4, 4, 4, cv::BORDER_CONSTANT, 0);

            cv::Mat imgFloat;
            square28.convertTo(imgFloat, CV_32F, 1.0f / 255.0f);
            const float* pred = nnPredict(model, imgFloat.ptr<float>(0), 784, nullptr);

            Prediction p;
            nnPredictedClasses(model, pred, 10, &p);
            if (p.confidence < 0.6f) {
                continue;
            }

            cv::rectangle(frame, { x, y }, { x + w, y + h }, { 0, 255, 0 }, 2);
            cv::putText(frame, std::to_string(p.cls), { x, y + h + 15 }, cv::FONT_HERSHEY_SIMPLEX, 0.5, { 255, 0, 0 });
        }

        cv::imshow("Video Feed", frame);

        if (cv::pollKey() == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();

    nnCleanup();
    return 0;
}
