//
// Created by minions on 10.03.16.
//

#ifndef IMAGE_PROCESSOR_OPENCV_MATCHING_HPP
#define IMAGE_PROCESSOR_OPENCV_MATCHING_HPP

#include <iostream>
#include <math.h>

#include <ros/ros.h>

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <eigen3/Eigen/Dense>

#define PI 3.14159265

namespace robotcam
{
    struct CurrentMatch {
        cv::Mat outFrame;
        std::vector <cv::Point2f> sceneCorners;
    };

    class OpenCVMatching
    {
    public:
        cv::Mat getCameraMatrix(const std::string path);

        cv::Mat getDistortionCoeff(const std::string path);

        std::string type2str(int type);

        cv::Mat captureFrame(bool color, bool undistort, cv::VideoCapture capture, cv::Mat cameraMatrix,
                             cv::Mat distCoeffs);

        cv::Mat captureFrame(bool color, cv::VideoCapture capture);

        std::vector <cv::DMatch> knnMatchDescriptors(cv::Mat &descriptors_object, cv::Mat &descriptors_scene,
                                                     float nnratio);

        std::vector <cv::DMatch> knnMatchDescriptorsLSH(cv::Mat &descriptors_object, cv::Mat &descriptors_scene,
                                                        float nndrRatio);

        std::vector <cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene);

        std::vector <cv::DMatch> bruteForce(cv::Mat &descriptors_object, cv::Mat &descriptors_scene, int normType);

        cv::Ptr <cv::Feature2D> setKeyPointsDetector(std::string typeKeyPoint);

        cv::Ptr <cv::Feature2D> setDescriptorsExtractor(std::string typeDescriptor, bool &binary);

        CurrentMatch visualizeMatch(cv::Mat &searchImage, cv::Mat &objectImage,
                                    std::vector <cv::KeyPoint> &keypointsObject,
                                    std::vector <cv::KeyPoint> &keypointsScene, std::vector <cv::DMatch> &good_matches,
                                    bool showKeypoints, int homographyType);

        bool checkObjectInnerAngles(std::vector <cv::Point2f> scorner, int min, int max);

        double getXoffset(cv::Mat frame, std::vector <cv::Point2f> scorner);

        double getYoffset(cv::Mat frame, std::vector <cv::Point2f> scorner);

        double getObjectAngle(cv::Mat frame, std::vector <cv::Point2f> scorner);

    private:
        bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);

        int innerAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c);

    };
}


#endif //IMAGE_PROCESSOR_OPENCV_MATCHING_HPP
