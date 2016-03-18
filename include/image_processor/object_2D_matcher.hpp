//
// Created by minions on 10.03.16.
//

#ifndef IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP
#define IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP

#include <geometry_msgs/Pose2D.h>

#include "image_processor/setProcessRunning.h"
#include "image_processor/getProcessRunning.h"
#include "image_processor/setBinaryMatching.h"
#include "image_processor/getBinaryMatching.h"
#include "image_processor/setKeypointDetectorType.h"
#include "image_processor/getKeypointDetectorType.h"
#include "image_processor/setDescriptorType.h"
#include "image_processor/getDescriptorType.h"
#include "image_processor/setVideoColor.h"
#include "image_processor/getVideoColor.h"
#include "image_processor/setBruteforceMatching.h"
#include "image_processor/getBruteforceMatching.h"
#include "image_processor/setVideoUndistortion.h"
#include "image_processor/getVideoUndistortion.h"

std::string DETECTOR_TYPE = "BRISK";
std::string EXTRACTOR_TYPE = "BRISK";

const std::string ref_path1 = "/home/asgeir/Desktop/ref_keypoints1.jpg";
const std::string ref_path2 = "/home/asgeir/Desktop/ref_keypoints2.jpg";

const int STEADYCAM_WIDTH = 1280;
const int STEADYCAM_HEIGHT = 720;

static const std::string OPENCV_WINDOW = "Matching";
const std::string CAMERA_PARAMS = "/home/asgeir/Documents/calibration_reserve_camera.yml";

std::string temp_path1, temp_path2;

geometry_msgs::Pose2D object_pose_msg;

int homographyMethod = CV_RANSAC;

void initializeMatcher(char **argv);

void detectAndComputeReference(cv::Mat &object, std::vector<cv::KeyPoint> &keypoints_object,
                               cv::Mat &descriptor_object);

void writeReferenceImage(cv::Mat object, std::vector<cv::KeyPoint> keypoints_object, std::string ref_path);

cv::Mat readImage(std::string path);

bool setProcessRunningCallBack(image_processor::setProcessRunning::Request &req,
                               image_processor::setProcessRunning::Response &res);

bool getProcessRunningCallBack(image_processor::getProcessRunning::Request &req,
                               image_processor::getProcessRunning::Response &res);

bool setBinaryMatchingCallBack(image_processor::setBinaryMatching::Request &req,
                               image_processor::setBinaryMatching::Response &res);

bool getBinaryMatchingCallBack(image_processor::getBinaryMatching::Request &req,
                               image_processor::getBinaryMatching::Response &res);

bool setBruteforceMatchingCallBack(image_processor::setBruteforceMatching::Request &req,
                                   image_processor::setBruteforceMatching::Response &res);

bool getBruteforceMatchingCallBack(image_processor::getBruteforceMatching::Request &req,
                                   image_processor::getBruteforceMatching::Response &res);

bool setKeypointDetectorTypeCallBack(image_processor::setKeypointDetectorType::Request &req,
                                     image_processor::setKeypointDetectorType::Response &res);

bool getKeypointDetectorTypeCallBack(image_processor::getKeypointDetectorType::Request &req,
                                     image_processor::getKeypointDetectorType::Response &res);

bool setDescriptorTypeCallBack(image_processor::setDescriptorType::Request &req,
                               image_processor::setDescriptorType::Response &res);

bool getDescriptorTypeCallBack(image_processor::getDescriptorType::Request &req,
                               image_processor::getDescriptorType::Response &res);

bool setVideoColorCallBack(image_processor::setVideoColor::Request &req,
                           image_processor::setVideoColor::Response &res);

bool getVideoColorCallBack(image_processor::getVideoColor::Request &req,
                           image_processor::getVideoColor::Response &res);

bool setVideoUndistortionCallBack(image_processor::setVideoUndistortion::Request &req,
                           image_processor::setVideoUndistortion::Response &res);

bool getVideoUndistortionCallBack(image_processor::getVideoUndistortion::Request &req,
                           image_processor::getVideoUndistortion::Response &res);


#endif //IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP
