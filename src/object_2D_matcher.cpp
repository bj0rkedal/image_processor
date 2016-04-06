//
// Created by minions on 10.03.16.
//
#include "../include/image_processor/openCV_matching.hpp"
#include "../include/image_processor/object_2D_matcher.hpp"

// Local variables
robotcam::OpenCVMatching openCVMatching;
robotcam::CurrentMatch match1;

// Video and reference images
cv::VideoCapture capture;
cv::Mat object1;

// Keypoints and descriptors
cv::Ptr<cv::Feature2D> detector, extractor;
std::vector<cv::KeyPoint> keypoints_object1, keypoints_scene;
cv::Mat descriptor_object1, descriptor_scene;

// Control
bool running = true;
bool binary = false;
bool bruteforce = true;
bool color = true;
bool undistort = true;

int main(int argc, char **argv) {
    ros::init(argc, argv, "object_2D_detection");
    ros::NodeHandle n;
    image_transport::ImageTransport it(n);
    // Topics
    image_transport::Publisher processed_pub = it.advertise("/object_2D_detected/image", 1);
    ros::Publisher pub1 = n.advertise<geometry_msgs::Pose2D>("/object_2D_detected/object1", 1);
    // Services
    ros::ServiceServer service1 = n.advertiseService("/object_2D_detection/setProcessRunning",
                                                     setProcessRunningCallBack);
    ros::ServiceServer service2 = n.advertiseService("/object_2D_detection/getProcessRunning",
                                                     getProcessRunningCallBack);
    ros::ServiceServer service3 = n.advertiseService("/object_2D_detection/setBinaryMatching",
                                                     setBinaryMatchingCallBack);
    ros::ServiceServer service4 = n.advertiseService("/object_2D_detection/getBinaryMatching",
                                                     getBinaryMatchingCallBack);
    ros::ServiceServer service5 = n.advertiseService("/object_2D_detection/setKeypointDetectorType",
                                                     setKeypointDetectorTypeCallBack);
    ros::ServiceServer service6 = n.advertiseService("/object_2D_detection/getKeypointDetectorType",
                                                     getKeypointDetectorTypeCallBack);
    ros::ServiceServer service7 = n.advertiseService("/object_2D_detection/setDescriptorType",
                                                     setDescriptorTypeCallBack);
    ros::ServiceServer service8 = n.advertiseService("/object_2D_detection/getDescriptorType",
                                                     getDescriptorTypeCallBack);
    ros::ServiceServer service9 = n.advertiseService("/object_2D_detection/setVideoColor",
                                                     setVideoColorCallBack);
    ros::ServiceServer service10 = n.advertiseService("/object_2D_detection/getVideoColor",
                                                      getVideoColorCallBack);
    ros::ServiceServer service11 = n.advertiseService("/object_2D_detection/setBruteforceMatching",
                                                      setBruteforceMatchingCallBack);
    ros::ServiceServer service12 = n.advertiseService("/object_2D_detection/getBruteforceMatching",
                                                      getBruteforceMatchingCallBack);
    ros::ServiceServer service13 = n.advertiseService("/object_2D_detection/setVideoUndistortion",
                                                      setVideoUndistortionCallBack);
    ros::ServiceServer service14 = n.advertiseService("/object_2D_detection/getVideoUndistortion",
                                                      getVideoUndistortionCallBack);
    ros::ServiceServer service15 = n.advertiseService("/object_2D_detection/setMatchingImage1",
                                                      setMatchingImage1CallBack);
    ros::Rate loop_rate(60);

    // Check camera
    if (!capture.open(0)) {
        ROS_ERROR(" --(!) Could not reach camera");
        return 0;
    }

    initializeMatcher(VIDEO_WIDTH,VIDEO_HEIGHT);

    // Check reference images
    if (!object1.data) {
        ROS_ERROR(" --(!) Error reading image");
        return 0;
    }
    ROS_INFO("Loaded reference image:\n\t%s", temp_path1.c_str());

    detectAndComputeReference(object1, keypoints_object1, descriptor_object1);
    writeReferenceImage(object1, keypoints_object1, ref_path1);

    cv::Mat cameraMatrix = openCVMatching.getCameraMatrix(CAMERA_PARAMS);
    cv::Mat distCoeffs = openCVMatching.getDistortionCoeff(CAMERA_PARAMS);

    sensor_msgs::ImagePtr image_msg;

    while (ros::ok()) {
        cv::Mat video = openCVMatching.captureFrame(color, undistort, capture, cameraMatrix, distCoeffs);
        if (video.empty()) break;

        // Uncomment if namedWindow and imshow is used
//        int key = 0xff & cv::waitKey(capture.isOpened() ? 30 : 500);
//        if( (key & 255) == 27 ) break;
        //

        if (running) {

            // Detect keypoints and descriptors
            detector->detect(video, keypoints_scene);
            extractor->compute(video, keypoints_scene, descriptor_scene);

            // Match descriptors of reference and video frame
            std::vector<cv::DMatch> good_matches;
            if (!binary) {
                if (bruteforce) {
                    good_matches = openCVMatching.bruteForce(descriptor_object1, descriptor_scene, cv::NORM_L1);
                } else {
                    good_matches = openCVMatching.knnMatchDescriptors(descriptor_object1, descriptor_scene, 0.9f);
                }
            } else {
                if (bruteforce) {
                    good_matches = openCVMatching.bruteForce(descriptor_object1, descriptor_scene, cv::NORM_HAMMING);
                } else {
                    good_matches = openCVMatching.knnMatchDescriptorsLSH(descriptor_object1, descriptor_scene, 0.9f);
                }
            }

            // Visualize matching
            if ((!keypoints_object1.size() == 0 && !keypoints_scene.size() == 0) && good_matches.size() >= 0) {

                match1 = openCVMatching.visualizedMatch(video, object1, keypoints_object1, keypoints_scene,
                                                        good_matches, true, CV_RANSAC);
                image_msg = cv_bridge::CvImage(
                        std_msgs::Header(), sensor_msgs::image_encodings::BGR8, match1.outFrame).toImageMsg();
                processed_pub.publish(image_msg);
                //cv::imshow(OPENCV_WINDOW, match2.outFrame);
            } else {
                image_msg = cv_bridge::CvImage(
                        std_msgs::Header(), sensor_msgs::image_encodings::BGR8, video).toImageMsg();
                processed_pub.publish(image_msg);
                //cv::imshow(OPENCV_WINDOW, video);
            }

        } else {
            image_msg = cv_bridge::CvImage(std_msgs::Header(), sensor_msgs::image_encodings::BGR8, video).toImageMsg();
            processed_pub.publish(image_msg);
            //cv::imshow(OPENCV_WINDOW, video);
        }

        // ROS
        if (match1.sceneCorners.size() == 4 && openCVMatching.checkObjectInnerAngles(match1.sceneCorners, 60, 120)) {
            double x = openCVMatching.getXpos(match1.sceneCorners);
            double y = openCVMatching.getYpos(match1.sceneCorners);

            object_pose_msg.theta = openCVMatching.getObjectAngle(video, match1.sceneCorners);

//            angleTest.push_back(object_pose_msg.theta);
//            int size = 20;
//            if(angleTest.size() == size) {
//                double d = 0.0;
//                for(int i = 0; i < angleTest.size(); i++) {
//                    d = d + angleTest.at(i);
//                    //std::cout << angleTest.at(i) << std::endl;
//                }
//                std::cout << d/size << std::endl;
//                angleTest.clear();
//            }

            object_pose_msg.x = x;
            object_pose_msg.y = y;
            pub1.publish(object_pose_msg);
        }

        ros::spinOnce();
        loop_rate.sleep();
    }
    ROS_INFO("Object detection shutting down");
    //cv::destroyWindow(OPENCV_WINDOW);
    return 0;
}

void initializeMatcher(const int video_width, const int video_height) {
    object1 = readImage(temp_path1);

    //cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_FULLSCREEN);

    capture.set(CV_CAP_PROP_FRAME_WIDTH, video_width);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, video_height);
    ROS_INFO("Camera resolution: width=%f, height=%f",
             capture.get(CV_CAP_PROP_FRAME_WIDTH),
             capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    detector = openCVMatching.setKeyPointsDetector(DETECTOR_TYPE);
    extractor = openCVMatching.setDescriptorsExtractor(EXTRACTOR_TYPE, binary);
    ROS_INFO("Bruteforce matching: %d", bruteforce);
}

void detectAndComputeReference(cv::Mat &object, std::vector<cv::KeyPoint> &keypoints_object,
                               cv::Mat &descriptor_object) {
    detector->detect(object, keypoints_object);
    extractor->compute(object, keypoints_object, descriptor_object);
}

void writeReferenceImage(cv::Mat object, std::vector<cv::KeyPoint> keypoints_object, std::string ref_path) {
    cv::Mat ref_keypoints;
    cv::drawKeypoints(object, keypoints_object, ref_keypoints, CV_RGB(255, 255, 0));
    cv::imwrite(ref_path, ref_keypoints);
    ROS_INFO("Reference keypoints written to: %s", ref_path.c_str());
}

cv::Mat readImage(std::string path) {
    cv::Mat object;
    if (color) {
        object = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    } else {
        object = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
    }
    return object;
}

bool setProcessRunningCallBack(image_processor::setProcessRunning::Request &req,
                               image_processor::setProcessRunning::Response &res) {
    running = req.running;
    return true;
}

bool getProcessRunningCallBack(image_processor::getProcessRunning::Request &req,
                               image_processor::getProcessRunning::Response &res) {
    res.running = running;
    return true;
}

bool setBinaryMatchingCallBack(image_processor::setBinaryMatching::Request &req,
                               image_processor::setBinaryMatching::Response &res) {
    binary = req.binary;
    return true;
}

bool getBinaryMatchingCallBack(image_processor::getBinaryMatching::Request &req,
                               image_processor::getBinaryMatching::Response &res) {
    res.binary = binary;
    return true;
}

bool setBruteforceMatchingCallBack(image_processor::setBruteforceMatching::Request &req,
                                   image_processor::setBruteforceMatching::Response &res) {
    bruteforce = req.bruteforce;
    return true;
}

bool getBruteforceMatchingCallBack(image_processor::getBruteforceMatching::Request &req,
                                   image_processor::getBruteforceMatching::Response &res) {
    res.bruteforce = bruteforce;
    return true;
}

bool setKeypointDetectorTypeCallBack(image_processor::setKeypointDetectorType::Request &req,
                                     image_processor::setKeypointDetectorType::Response &res) {
    DETECTOR_TYPE = req.type;
    detector = openCVMatching.setKeyPointsDetector(DETECTOR_TYPE);
    detector->detect(object1, keypoints_object1);
    writeReferenceImage(object1, keypoints_object1, ref_path1);
    return true;
}

bool getKeypointDetectorTypeCallBack(image_processor::getKeypointDetectorType::Request &req,
                                     image_processor::getKeypointDetectorType::Response &res) {
    res.type = DETECTOR_TYPE;
    return true;
}

bool setDescriptorTypeCallBack(image_processor::setDescriptorType::Request &req,
                               image_processor::setDescriptorType::Response &res) {
    EXTRACTOR_TYPE = req.type;
    extractor = openCVMatching.setDescriptorsExtractor(EXTRACTOR_TYPE, binary);
    extractor->compute(object1, keypoints_object1, descriptor_object1);
    return true;
}

bool getDescriptorTypeCallBack(image_processor::getDescriptorType::Request &req,
                               image_processor::getDescriptorType::Response &res) {
    res.type = EXTRACTOR_TYPE;
    return true;
}

bool setVideoColorCallBack(image_processor::setVideoColor::Request &req,
                           image_processor::setVideoColor::Response &res) {
    color = req.color;
    object1 = readImage(temp_path1);
    keypoints_object1.clear();
    descriptor_object1.release();
    detectAndComputeReference(object1, keypoints_object1, descriptor_object1);
    writeReferenceImage(object1, keypoints_object1, ref_path1);
    return true;
}

bool getVideoColorCallBack(image_processor::getVideoColor::Request &req,
                           image_processor::getVideoColor::Response &res) {
    res.color = color;
    return true;
}

bool setVideoUndistortionCallBack(image_processor::setVideoUndistortion::Request &req,
                                  image_processor::setVideoUndistortion::Response &res) {
    undistort = req.undistort;
    return true;
}

bool getVideoUndistortionCallBack(image_processor::getVideoUndistortion::Request &req,
                                  image_processor::getVideoUndistortion::Response &res) {
    res.undistort = undistort;
    return true;
}

bool setMatchingImage1CallBack(image_processor::setMatchingImage1::Request &req,
                               image_processor::setMatchingImage1::Response &res) {
    temp_path1 = req.imagePath;
    object1 = readImage(temp_path1);
    keypoints_object1.clear();
    descriptor_object1.release();
    detectAndComputeReference(object1, keypoints_object1, descriptor_object1);
    writeReferenceImage(object1, keypoints_object1, ref_path1);
    return true;
}