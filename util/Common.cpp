#include "Common.h"


cv::Mat load(const string& img_file_name, int mode){
    /*
        input:
            Input image file name
            mode = 1(BGR) || 0(Gray scale)
        output:
            image Mat
    */
    Mat img= imread(img_file_name, mode);
    if (img.empty()){
        CV_Error(-2, "\nImage fail to read!\n");
        return Mat();
    }
    return img;
}

Mat binary(Mat& input_img, int THREHOLD){
    /*
        input: 
            Gray scale Mat 
            Threshold value
        output: 
            hard-binarized Mat
    */
    if(input_img.empty()){
        CV_Error(-2, "\nInput image is empty!\n");
        return Mat();
    }
    if(input_img.channels() > 1){
        CV_Error(-2, "\nTo binarize an image, input image needs to be gray scale\n");
        return Mat();
    }
    int rows = input_img.rows;
    int cols = input_img.cols;
    Mat bin_img(rows, cols, CV_8UC1, Scalar(255));
    for(int i =0; i< rows;++i){
        for (int j =0;j< cols;++j){
            if (input_img.at<uchar>(i, j) < THREHOLD){
                bin_img.at<uchar>(i, j) = 0;
            }
        }// end cols
    }// end rows
    return bin_img;

}