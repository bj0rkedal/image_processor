//
// Created by Asgeir Bjørkedal on 10.03.16.
// Part of a vision solution for a master's thesis.
//

#ifndef IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP
#define IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP

#include <ros/package.h>

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
#include "image_processor/setMatchingImage1.h"
#include "image_processor/setImageDepth.h"

#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// Keypoint and descriptor type
std::string DETECTOR_TYPE = "SIFT";
std::string EXTRACTOR_TYPE = "SIFT";

// Window name
static const std::string OPENCV_WINDOW = "Matching";

// Resolution
const int VIDEO_WIDTH = 1280;
const int VIDEO_HEIGHT = 720;

// Path to camera parameters (K-matrix)
const std::string CAMERA_PARAMS = ros::package::getPath("image_processor")
                                  + "/resources/calibration_reserve_camera.yml";

// Path to reference image storage
const std::string ref_path1 = ros::package::getPath("image_processor")
                              + "/resources/output/ref_keypoints1.jpg";

// Path to initial matching image
std::string temp_path1 = ros::package::getPath("image_processor")
                         + "/resources/Lenna.png";

// Holds the object pose
geometry_msgs::Pose2D object_pose_msg;

// Homography method
int homographyMethod = CV_RANSAC; // CV_LMEDS

// Loop frequency
double FREQ = 60;

/*!
 * \brief Initializes the object matcher image, resolution, detector and extractor.
 * \param video_width The horizontal video resolution (pixel)
 * \param video_height The vertical video resolution (pixel)
 */
void initializeMatcher(const int video_width, const int video_height);

/*!
 * \brief Detect and compute keypoints and descriptors for a given image matrix.
 * \param object The image
 * \param keypoints_object Reference to the keypoints storage object
 * \param descriptor_object Reference to the descriptor storage object
 */
void detectAndComputeReference(cv::Mat &object, std::vector<cv::KeyPoint> &keypoints_object,
                               cv::Mat &descriptor_object);

/*!
 * \brief Draws keypoints on a chosen image object and stores it to a desired file path.
 * \param object The image
 * \param keypoints_object The keypoints
 * \param ref_path The storage file path
 */
void writeReferenceImage(cv::Mat object, std::vector<cv::KeyPoint> keypoints_object, std::string ref_path);

/*!
 * \brief Read an image from a desired file path.
 * \param path The file path
 * \return The image matrix
 */
cv::Mat readImage(std::string path);

/*!
 * \brief Callback method for toggling the object detection through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setProcessRunningCallBack(image_processor::setProcessRunning::Request &req,
                               image_processor::setProcessRunning::Response &res);

/*!
 * \brief Callback method for object detection running status through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getProcessRunningCallBack(image_processor::getProcessRunning::Request &req,
                               image_processor::getProcessRunning::Response &res);

/*!
 * \brief Callback method for toggling of binary/non-binary matching through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setBinaryMatchingCallBack(image_processor::setBinaryMatching::Request &req,
                               image_processor::setBinaryMatching::Response &res);

/*!
 * \brief Callback method for binary/non-binary matching status through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getBinaryMatchingCallBack(image_processor::getBinaryMatching::Request &req,
                               image_processor::getBinaryMatching::Response &res);

/*!
 * \brief Callback method for toggling between bruteforce and FLANN matching through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setBruteforceMatchingCallBack(image_processor::setBruteforceMatching::Request &req,
                                   image_processor::setBruteforceMatching::Response &res);

/*!
 * \brief Callback method for bruteforce/FLANN matching status through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getBruteforceMatchingCallBack(image_processor::getBruteforceMatching::Request &req,
                                   image_processor::getBruteforceMatching::Response &res);

/*!
 * \brief Callback method for setting keypoint detector through ROS service.
 * Sets the detector based on a string input. Detects keypoints in the matching image and outputs an image file with
 * the new keypoints.
 * \param req The service request
 * \param res The service response
 */
bool setKeypointDetectorTypeCallBack(image_processor::setKeypointDetectorType::Request &req,
                                     image_processor::setKeypointDetectorType::Response &res);

/*!
 * \brief Callback method for getting the keypoint detector type through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getKeypointDetectorTypeCallBack(image_processor::getKeypointDetectorType::Request &req,
                                     image_processor::getKeypointDetectorType::Response &res);

/*!
 * \brief Callback method for setting descriptor extractor through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setDescriptorTypeCallBack(image_processor::setDescriptorType::Request &req,
                               image_processor::setDescriptorType::Response &res);

/*!
 * \brief Callback method for getting descriptor extractor type through ROS service.
 * Sets the extractor based on a string input. New descriptors are computed for the matching image.
 * Further matching with the new descriptor can be performed instantanously.
 * \param req The service request
 * \param res The service response
 */
bool getDescriptorTypeCallBack(image_processor::getDescriptorType::Request &req,
                               image_processor::getDescriptorType::Response &res);

/*!
 * \brief Callback method for setting color/grayscale video capture through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setVideoColorCallBack(image_processor::setVideoColor::Request &req,
                           image_processor::setVideoColor::Response &res);

/*!
 * \brief Callback method for getting current video color mode through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getVideoColorCallBack(image_processor::getVideoColor::Request &req,
                           image_processor::getVideoColor::Response &res);

/*!
 * \brief Callback method for setting video undistortion of video stream through ROS service.
 * Undistortion will use distortion parameters from .XML/.YAML file output from camera calibration.
 * \param req The service request
 * \param res The service response
 */
bool setVideoUndistortionCallBack(image_processor::setVideoUndistortion::Request &req,
                                  image_processor::setVideoUndistortion::Response &res);

/*!
 * \brief Callback method for getting undistortion status through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool getVideoUndistortionCallBack(image_processor::getVideoUndistortion::Request &req,
                                  image_processor::getVideoUndistortion::Response &res);

/*!
 * \brief Callback method for setting the image to match with in the video scene through ROS service.
 * Reads the new image, detects keypoints and computes descriptors, and outputs an image with keypoints.
 * \param req The service request
 * \param res The service response
 */
bool setMatchingImage1CallBack(image_processor::setMatchingImage1::Request &req,
                               image_processor::setMatchingImage1::Response &res);

/*!
 * \brief Callback method for setting the image depth (lambda), used for scaling the normalized image coordinates
 * through ROS service.
 * \param req The service request
 * \param res The service response
 */
bool setImageDepthCallBack(image_processor::setImageDepth::Request &req,
                           image_processor::setImageDepth::Response &res);


#endif //IMAGE_PROCESSOR_OBJECT_2D_MATCHER_HPP
