//
// Created by minions on 10.03.16.
//

#include "../include/image_processor/openCV_matching.hpp"

namespace robotcam
{
    cv::Mat OpenCVMatching::getCameraMatrix(const std::string path) {
        cv::Mat temp;
        cv::FileStorage fs(path, cv::FileStorage::READ);
        fs["camera_matrix"] >> temp;
        fs.release();

        return temp;
    }

    cv::Mat OpenCVMatching::getDistortionCoeff(const std::string path) {
        cv::Mat temp;
        cv::FileStorage fs(path, cv::FileStorage::READ);
        fs["distortion_coefficients"] >> temp;
        fs.release();

        return temp;
    }

    std::string OpenCVMatching::type2str(int type) {
        std::string r;

        uchar depth = type & CV_MAT_DEPTH_MASK;
        uchar chans = 1 + (type >> CV_CN_SHIFT);

        switch (depth) {
            case CV_8U:
                r = "8U";
                break;
            case CV_8S:
                r = "8S";
                break;
            case CV_16U:
                r = "16U";
                break;
            case CV_16S:
                r = "16S";
                break;
            case CV_32S:
                r = "32S";
                break;
            case CV_32F:
                r = "32F";
                break;
            case CV_64F:
                r = "64F";
                break;
            default:
                r = "User";
                break;
        }

        r += "C";
        r += (chans + '0');

        // USAGE
        //    std::string ty =  type2str( H.type() );
        //    printf("Matrix: %s %dx%d \n", ty.c_str(), H.cols, H.rows );

        return r;
    }

    cv::Mat OpenCVMatching::captureFrame(bool color, bool useCalibration, cv::VideoCapture capture,
                                         cv::Mat cameraMatrix, cv::Mat distCoeffs) {
        cv::Mat inFrame, outFrame;
        capture >> inFrame;

        // Check if frame should be color or grayscale
        if (color == false && useCalibration == false) {
            cv::cvtColor(inFrame, outFrame, CV_RGB2GRAY); // grayscale
        } else if (color == false && useCalibration == true) {
            cv::Mat temp;
            cv::undistort(inFrame, temp, cameraMatrix, distCoeffs);
            cv::cvtColor(temp, outFrame, CV_RGB2GRAY); // grayscale
        } else if (color == true && useCalibration == false) {
            outFrame = inFrame;
        } else {
            cv::undistort(inFrame, outFrame, cameraMatrix, distCoeffs);
        }

        return outFrame;
    }

    cv::Mat OpenCVMatching::captureFrame(bool color, cv::VideoCapture capture) {
        cv::Mat inFrame, outFrame;
        capture >> inFrame;
        if(color) {
            outFrame = inFrame;
        } else {
            cv:cvtColor(inFrame, outFrame, CV_RGB2GRAY);
        }
        return outFrame;
    }

    std::vector<cv::DMatch> OpenCVMatching::knnMatchDescriptors(cv::Mat &descriptors_object, cv::Mat &descriptors_scene,
                                                                float nnratio) {
        cv::FlannBasedMatcher matcher;
        std::vector<std::vector<cv::DMatch> > matches;

        // Match descriptors
        matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);

        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());

        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;

            const cv::DMatch &m1 = matches[i][0];
            const cv::DMatch &m2 = matches[i][1];

            if (m1.distance <= nnratio * m2.distance)
                good_matches.push_back(m1);
        }

        return good_matches;
    }

    std::vector<cv::DMatch> OpenCVMatching::knnMatchDescriptorsLSH(cv::Mat &descriptors_object,
                                                                   cv::Mat &descriptors_scene, float nndrRatio) {
        cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
        std::vector<std::vector<cv::DMatch> > matches;

        // Match descriptors
        matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);

        std::vector<cv::DMatch> good_matches;
        good_matches.reserve(matches.size());

        for (size_t i = 0; i < matches.size(); ++i) {
            if (matches[i].size() < 2)
                continue;

            const cv::DMatch &m1 = matches[i][0];
            const cv::DMatch &m2 = matches[i][1];

            if (m1.distance <= nndrRatio * m2.distance)
                good_matches.push_back(m1);
        }

        return good_matches;
    }

    std::vector<cv::DMatch> OpenCVMatching::matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene) {
        // Good at filtering FLANN based match.
        cv::FlannBasedMatcher matcher;
        std::vector<cv::DMatch> matches;

        // Match descriptors
        matcher.match(descriptors_object, descriptors_scene, matches);

        double max_dist = 0;
        double min_dist = 100;

        // Compute the max and min distance of the matches in current videoFrame
        for (int i = 0; i < descriptors_object.rows; i++) {
            double dist = matches[i].distance;
            if (dist < min_dist) min_dist = dist;
            if (dist > max_dist) max_dist = dist;
        }

        std::vector<cv::DMatch> good_matches;

        // Remove bad matches
        double k = 2;
        for (int i = 0; i < descriptors_object.rows; i++) {
            if (matches[i].distance <= cv::max(k * min_dist, 0.02)) {
                good_matches.push_back(matches[i]);
            }
        }

        return good_matches;
    }

    std::vector<cv::DMatch> OpenCVMatching::bruteForce(cv::Mat &descriptors_object, cv::Mat &descriptors_scene,
                                                       int normType) {
        cv::BFMatcher matcher(normType);
        std::vector<std::vector<cv::DMatch> > matches;
        matcher.knnMatch(descriptors_object, descriptors_scene, matches, 2);

//    look whether the match is inside a defined area of the image
//    only 25% of maximum of possible distance
//    double tresholdDist = 0.25 * sqrt(double(object.size().height*object.size().height
//                                             +object.size().width*object.size().width));

        std::vector<cv::DMatch> good_matches;
        for (int i = 0; i < matches.size(); ++i) {
            const float ratio = 0.9; // As in Lowe's paper; can be tuned
            if (matches[i][0].distance < ratio * matches[i][1].distance) {
                good_matches.push_back(matches[i][0]);
            }
        }

        return good_matches;
    }

    cv::Ptr<cv::Feature2D> OpenCVMatching::setKeyPointsDetector(std::string typeKeyPoint) {
        cv::Ptr<cv::Feature2D> detector;
        if (typeKeyPoint == "SURF") {
            detector = cv::xfeatures2d::SURF::create(1000,4,5,false,false);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "SIFT") {
            detector = cv::xfeatures2d::SIFT::create(0,5,0.04,10,1.6);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "STAR") {
            detector = cv::xfeatures2d::StarDetector::create(45,30,10,8,5);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "BRISK") {
            detector = cv::BRISK::create(30,3,1.0f);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "FAST") {
            detector = cv::FastFeatureDetector::create(10,true,cv::FastFeatureDetector::TYPE_9_16);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "ORB") {
            detector = cv::ORB::create(1000,1.2f,8,31,0,2,cv::ORB::FAST_SCORE,31,20);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else if (typeKeyPoint == "AKAZE") {
            detector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,4,4,cv::KAZE::DIFF_PM_G2);
            ROS_INFO("Keypoint detector: %s", typeKeyPoint.c_str());
        } else {
            ROS_ERROR("Could not find keypoint detector: %s\n\tChoosing default: SURF", typeKeyPoint.c_str());
            detector = cv::xfeatures2d::SURF::create(1000);
        }
        return detector;
    }

    cv::Ptr<cv::Feature2D> OpenCVMatching::setDescriptorsExtractor(std::string typeDescriptor, bool &binary) {
        cv::Ptr<cv::Feature2D> extractor;
        if (typeDescriptor == "SURF") {
            binary = false;
            extractor = cv::xfeatures2d::SURF::create(1000,4,5,false,false);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "SIFT") {
            binary = false;
            extractor = cv::xfeatures2d::SIFT::create(0,5,0.04,10,1.6);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "BRISK") {
            binary = true;
            extractor = cv::BRISK::create(30,3,1.0f);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "FREAK") {
            binary = true;
            extractor = cv::xfeatures2d::FREAK::create(true,true,22.0f,4);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "ORB") {
            binary = true;
            extractor = cv::ORB::create(1000,1.2f,8,31,0,2,cv::ORB::FAST_SCORE,31,20); // WTA_K = 3-4 -> HAMMING2
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "AKAZE") {
            binary = true;
            extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,4,4,cv::KAZE::DIFF_PM_G2);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else if (typeDescriptor == "BRIEF") {
            binary = true;
            extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
            ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        } else {
            binary = false;
            ROS_ERROR("Could not find keypoint detector: %s\n\tChoosing default descriptor: SURF", typeDescriptor.c_str());
            extractor = cv::xfeatures2d::SURF::create(1000);
        }
        return extractor;
    }

    CurrentMatch OpenCVMatching::visualizeMatch(cv::Mat &searchImage, cv::Mat &objectImage,
                                                std::vector<cv::KeyPoint> &keypointsObject,
                                                std::vector<cv::KeyPoint> &keypointsScene,
                                                std::vector<cv::DMatch> &good_matches,
                                                bool showKeypoints, int homographyType) {

        cv::Mat image_matches;

        if (showKeypoints) {
            cv::drawKeypoints(searchImage, keypointsScene, image_matches, CV_RGB(255,255,0),
                              cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS); // cv::Scalar::all(-1)
//        cv::drawMatches(objectImage, keypointsObject, searchImage, keypointsScene, good_matches, image_matches,
//                        cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
//                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        } else {
            image_matches = searchImage.clone();
        }

        std::vector<cv::Point2f> obj;
        std::vector<cv::Point2f> scene;

        for (size_t i = 0; i < good_matches.size(); i++) {
            // Retrieve the keypoints from good matches
            obj.push_back(keypointsObject[good_matches[i].queryIdx].pt);
            scene.push_back(keypointsScene[good_matches[i].trainIdx].pt);
        }

        // Perform Homography to find a perspective transformation between two planes.
        cv::Mat H;
        if (!obj.size() == 0 && !scene.size() == 0) {
            H = cv::findHomography(obj, scene, homographyType); // CV_LMEDS // CV_RANSAC
        }

        std::vector<cv::Point2f> objectCorners(4);

        // Put object corners in a vector
        objectCorners[0] = cvPoint(0, 0); //Upper left corner
        objectCorners[1] = cvPoint(objectImage.cols, 0); //Upper right corner
        objectCorners[2] = cvPoint(objectImage.cols, objectImage.rows); //Lower right corner
        objectCorners[3] = cvPoint(0, objectImage.rows); //Lower left corner

        std::vector<cv::Point2f> sceneCorners(4);

        // Find the object corners in the scene perspective
        if (!H.rows == 0 && !H.cols == 0) {
            cv::perspectiveTransform(objectCorners, sceneCorners, H);

            if (checkObjectInnerAngles(sceneCorners, 60, 120)) {
                // Draw lines surrounding the object
                cv::line(image_matches, sceneCorners[0], sceneCorners[1], cv::Scalar(0, 255, 0), 2); //TOP line
                cv::line(image_matches, sceneCorners[1], sceneCorners[2], cv::Scalar(0, 255, 0), 2); //RIGHT line
                cv::line(image_matches, sceneCorners[2], sceneCorners[3], cv::Scalar(0, 255, 0), 2); //BOTTOM line
                cv::line(image_matches, sceneCorners[3], sceneCorners[0], cv::Scalar(0, 255, 0), 2); //LEFT line
                // Draw diagonals
                cv::line(image_matches, sceneCorners[0], sceneCorners[2], cv::Scalar(0, 255, 0), 1); //DIAGONAL 0-2
                cv::line(image_matches, sceneCorners[1], sceneCorners[3], cv::Scalar(0, 255, 0), 1); //DIAGONAL 1-3

                // Center
                cv::Point2f cen(0.0, 0.0);
                if (intersection(sceneCorners[0], sceneCorners[2], sceneCorners[1], sceneCorners[3], cen)) {
                    cv::circle(image_matches, cen, 10, cv::Scalar(0, 0, 255), 2);
                }
            }
        }

        // Draw circles in center pixel of the video stream
        if (searchImage.rows > 60 && searchImage.cols > 60) {
            cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 5, CV_RGB(255, 0, 0));
            cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 10, CV_RGB(0, 255, 0));
            cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 15, CV_RGB(0, 0, 255));
        }

        CurrentMatch cm;
        cm.outFrame = image_matches;
        cm.sceneCorners = sceneCorners;

        return cm;
    }

    bool OpenCVMatching::intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r) {
        // The lines are defined by (o1, p1) and (o2, p2).
        cv::Point2f x = o2 - o1;
        cv::Point2f d1 = p1 - o1;
        cv::Point2f d2 = p2 - o2;

        float cross = d1.x * d2.y - d1.y * d2.x;
        if (fabsf(cross) < /*EPS*/1e-8)
            return false;

        double t1 = (x.x * d2.y - x.y * d2.x) / cross;
        r = o1 + d1 * t1;
        return true;
    }

    int OpenCVMatching::innerAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
        cv::Point2f ab(b.x - a.x, b.y - a.y);
        cv::Point2f cb(b.x - c.x, b.y - c.y);

        double dot = (ab.x * cb.x + ab.y * cb.y); // dot product
        double cross = (ab.x * cb.y - ab.y * cb.x); // cross product

        double alpha = atan2(cross, dot);

        int angle = (int) floor(alpha * 180. / PI + 0.5);
        //alpha = alpha * 180 / PI;

        return abs(angle);
    }

    bool OpenCVMatching::checkObjectInnerAngles(std::vector<cv::Point2f> scorner, int min, int max) {
        bool out = false;
        int c0 = innerAngle(scorner[3], scorner[0], scorner[1]);
        int c1 = innerAngle(scorner[0], scorner[1], scorner[2]);
        int c2 = innerAngle(scorner[1], scorner[2], scorner[3]);
        int c3 = innerAngle(scorner[2], scorner[3], scorner[0]);

        if (c0 > min && c0 < max && c1 > min && c1 < max && c2 > min && c2 < max && c3 > min && c3 < max) out = true;

        return out;
    }

    double OpenCVMatching::getXoffset(cv::Mat frame, std::vector<cv::Point2f> scorner) {
        cv::Point2f cen;
        double xOffset = 0.0;

        if (intersection(scorner[0], scorner[2], scorner[1], scorner[3], cen)) {
            xOffset = cen.x - frame.cols / 2;
        }

        return xOffset;
    }

    double OpenCVMatching::getYoffset(cv::Mat frame, std::vector<cv::Point2f> scorner) {
        cv::Point2f cen;
        double yOffset = 0.0;

        if (intersection(scorner[0], scorner[2], scorner[1], scorner[3], cen)) {
            yOffset = cen.y - frame.rows / 2;
        }

        return yOffset;
    }

    double OpenCVMatching::getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner) {

        double centerX = frame.cols / 2;
        double diffX = centerX - scorner[1].x;
        double x = (centerX - diffX) - scorner[0].x;

        double y = scorner[0].y - scorner[1].y;

        double angle = atan2(y, x) * 180 / PI;
        return angle;
    }
}