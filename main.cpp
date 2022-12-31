#include <opencv2/opencv.hpp>
#include "util/Common.h"
#include <iostream>
#include <vector>
#include <algorithm> //std::sort
#include <map>
using std::cout;
using std::endl;
using std::vector;
using std::map;
using namespace cv;
vector<vector<Point> > extractDigitCard(Mat gray){
    Mat bin = binary(gray, 132);
    vector<vector<Point> > contours;
    vector<vector<Point> > filterd_contours;
    cv::findContours(bin, contours, 3, 1);
    for(int i =0;i< contours.size();i++){
        cv::Rect bbox = cv::boundingRect(contours[i]);
        double height = bbox.height;
        double width = bbox.width;
        double ratio = width / height;
        if (ratio > 0.5 && ratio < 0.75 && height > 20){ // hand tuned 
            filterd_contours.push_back(contours[i]);
        }
    }// end for contour
    return filterd_contours;
}

vector<vector<Point> > extractDigitTemplate(Mat gray){
    vector<vector<Point> > contours, filtered_contours;
    Mat bin = binary(gray, 150);
    cv::findContours(bin, contours, 3, 1);
    for(int i =0;i<contours.size();i++){
        cv::Rect bbox = cv::boundingRect(contours[i]);
        double height = bbox.height, width = bbox.width;
        if(height > 50 && height < 60){
            filtered_contours.push_back(contours[i]);
        }
    }
    return filtered_contours;
}

bool compareControuCoordinate(const vector<Point> &p1, const vector<Point> &p2){
    return (p1[0].x < p2[0].x);
}
map<int, Mat> extractRoi(const vector<vector<Point>> &contour_template, Mat gray){
    map<int , Mat> map;
    for(int i =0;i<contour_template.size();++i){
        cv::Rect bbox= cv::boundingRect(contour_template[i]);
        // select a slightly large roi
        // bbox.x-=5;
        // bbox.y-=5;
        // bbox.width+=5;
        // bbox.height+=5;
        Mat roi = gray(bbox);
        Mat resized;
        cv::resize(roi, resized,Size(50, 50));
        map.insert(std::make_pair(i, resized));
    }
    return map;
}
int main(int argc, char** argv){
    Mat gray_credit_card = load("/Users/haochenyu/Desktop/CreditCardReader/ocr_input.jpg", 0);
    Mat color_credit_card = load("/Users/haochenyu/Desktop/CreditCardReader/ocr_input.jpg", 1);

    Mat gray_template = load("/Users/haochenyu/Desktop/CreditCardReader/ocr_a_reference_bitwise.jpg",0);
    Mat color_template = load("/Users/haochenyu/Desktop/CreditCardReader/ocr_a_reference_bitwise.jpg", 1);
    
    // extract digits for template
    vector<vector<Point> > contour_template = extractDigitTemplate(gray_template);

    // sort contour;
    std::sort(contour_template.begin(), contour_template.end(), compareControuCoordinate);

    // put into a hashtable
    map<int, Mat> map = extractRoi(contour_template, gray_template);
    



    // extract digits for credit card
    vector<vector<Point> > contour_card = extractDigitCard(gray_credit_card);
    std::sort(contour_card.begin(), contour_card.end(), compareControuCoordinate);

    // detect!
    string digits = "";
    for(int i =0;i<contour_card.size();++i){
        cv::Rect bbox= cv::boundingRect(contour_card[i]);
        Mat bin = binary(gray_credit_card, 132);
        Mat roi = bin(bbox);

        Mat resized_card;
        cv::resize(roi, resized_card,Size(50, 50));
        
        Mat result_mat(10, 1, CV_64FC1, Scalar(0));
        for(const auto& p: map){
            Mat Template = p.second;
            Mat diff = resized_card - Template;
            double n = (double) diff.total(); // number of pixels column * channel * rows;
            double mse = diff.dot(diff.t()) / n;
            result_mat.at<double>(p.first, 0) = mse;
        }
        double minVal, maxVal;
        Point minLoc, maxLoc;
        cv::minMaxLoc(result_mat, &minVal, &maxVal, &minLoc, &maxLoc);
        int digit = minLoc.y;
        if (i % 4 == 0){
            digits += ' ';
        }
        digits += char(digit + '0');
    }
    
    cv::putText(color_credit_card,digits,Point(70, 50),1, 2, Scalar(0,0,255),2);
    cv::imwrite("detection_result.png", color_credit_card.clone());
    imshow("Detect digits", color_credit_card);
    waitKey(0);
}