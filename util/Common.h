#ifndef COMMON_H
#define COMMON_H

#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;
using std::string;


// load image
cv::Mat load(const string&, int);

// binarize a gray scale image
cv::Mat binary(cv::Mat&, int);


#endif