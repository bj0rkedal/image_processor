//
// Created by minions on 17.02.16.
//

#ifndef TEST_OPENCV_OPENCV_TESTER_HPP
#define TEST_OPENCV_OPENCV_TESTER_HPP

#include <iostream>
#include <math.h>

#include <ros/ros.h>
#include <geometry_msgs/Pose2D.h>

#include "image_processor/setProcessRunning.h"
#include "image_processor/getProcessRunning.h"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/xfeatures2d.hpp"

#include <eigen3/Eigen/Dense>

#define PI 3.14159265

namespace robotCam {

    class openCV_tester {
    public:
        openCV_tester();

        ~openCV_tester();

        cv::Mat getVideoFrame();

    private:
    };
}

struct CurrentMatch{
    cv::Mat outFrame;
    std::vector<cv::Point2f> sceneCorners;
};

// Window
static const std::string OPENCV_WINDOW = "Matching";

// Video
cv::VideoCapture capture;
cv::Mat videoFrame;
int resWidth = 1280;
int resHeight = 720;
cv::Mat imagetest;

// Calibration
cv::Mat cameraMatrix, distCoeffs;
const std::string CALIBRATION_FILE = "/home/minions/calibration_gimbal_720p.yml";

// Reference image
const std::string ref_path1 = "/home/minions/Pictures/opencvtest/ref_keypoints1.jpg";
const std::string ref_path2 = "/home/minions/Pictures/opencvtest/ref_keypoints2.jpg";
cv::Mat object_image, object_image2;
cv::Mat ref_keypoints1, ref_keypoints2;

// Feature computation
double minHessian = 400;
int nOctaves=4;
int nOctaveLayers=5;
bool extended=false; //true betyr 128 element per descriptor. false = 64
bool upright=false; //true betyr at orientering av features blir skippet.
cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SURF::create(minHessian,nOctaves,nOctaveLayers,extended,upright);

//int nFeatures = 0;
//int nOctaveLayers = 5;
//double contrastThreshold = 0.04;
//double edgeThreshold = 10;
//double sigma = 1.6;
//cv::Ptr<cv::Feature2D> f2d = cv::xfeatures2d::SIFT::create(nFeatures,nOctaveLayers,contrastThreshold,
//                                                           edgeThreshold,sigma);

std::vector<cv::KeyPoint> ko, ks, ko2;
cv::Mat deso, dess, deso2;

CurrentMatch match1;
CurrentMatch match2;

// ROS
double loop_frequency = 60;
geometry_msgs::Pose2D pose_msg;

// Service variables
bool running = false;

// Methods
cv::Mat getCameraMatrix(const std::string path);

cv::Mat getDistortionCoeff(const std::string path);

std::string type2str(int type);

/*! \brief Capture a frame from a connected web camera.
 *  \param color The boolean determining RGB or grayscale video frame.
 *  \param undistort The boolean determining calibration of camera.
 *  \param capture The object capturing the video stream from the camera.
 *
 *  The method requires three arguments to determine the output video frame.
 */
cv::Mat captureFrame(bool color, bool undistort, cv::VideoCapture capture);

std::vector<cv::DMatch> knnMatchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nndrRatio);

std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene);

CurrentMatch visualizeMatch(cv::Mat searchImage, cv::Mat objectImage, std::vector<cv::KeyPoint> keypointsObject,
                            std::vector<cv::KeyPoint> keypointsScene, std::vector<cv::DMatch> good_matches,
                            bool showKeypoints);

bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);

int innerAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c);

bool checkObjectInnerAngles(std::vector<cv::Point2f> scorner, int min, int max);

cv::Point2f getObjectCentroidMean(std::vector<cv::Point2f> scorner);

cv::Point2f getObjectCentroidBbox(std::vector<cv::Point2f> scorner);

double getXoffset(cv::Mat frame, std::vector<cv::Point2f> scorner);

double getYoffset(cv::Mat frame, std::vector<cv::Point2f> scorner);

double getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner);

bool setProcessRunningCallBack(image_processor::setProcessRunning::Request &req,
                               image_processor::setProcessRunning::Response &res);

bool getProcessRunningCallBack(image_processor::getProcessRunning::Request &req,
                               image_processor::getProcessRunning::Response &res);

#endif //TEST_OPENCV_OPENCV_TESTER_HPP
