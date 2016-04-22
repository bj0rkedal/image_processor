//
// Created by Asgeir Bjørkedal on 17.02.16.
// Test code
//

#include "../include/image_processor/openCV_tester.hpp"

namespace robotCam {
    // Constructor
    openCV_tester::openCV_tester() { }

    // Deconstructor
    openCV_tester::~openCV_tester() { }
}

enum {SURF = 1, SIFT = 2, STAR = 3, FAST = 4, BRISK = 5, AKAZE = 6, ORB = 7, FREAK = 8, BRIEF = 9};

cv::Mat getCameraMatrix(const std::string path) {
    cv::Mat temp;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["camera_matrix"] >> temp;
    fs.release();

    return temp;
}

cv::Mat getDistortionCoeff(const std::string path) {
    cv::Mat temp;
    cv::FileStorage fs(path, cv::FileStorage::READ);
    fs["distortion_coefficients"] >> temp;
    fs.release();

    return temp;
}

std::string type2str(int type) {
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

cv::Mat captureFrame(bool color, bool useCalibration, cv::VideoCapture capture) {
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

std::vector<cv::DMatch> knnMatchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nnratio) {
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

std::vector<cv::DMatch> knnMatchDescriptorsLSH(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nndrRatio) {
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

std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene) {
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

std::vector<cv::DMatch> bruteForce(cv::Mat descriptors_object, cv::Mat descriptors_scene, int normType) {
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

cv::Ptr<cv::Feature2D> getKeypoints(int detectorType) {
    cv::Ptr<cv::Feature2D> keyPointDetector;
    switch(detectorType) {
        case SURF:
            keyPointDetector = cv::xfeatures2d::SURF::create(1000,4,5,false,false);
            ROS_INFO("Keypoint detector: SURF");
            return keyPointDetector;
        case SIFT:
            keyPointDetector = cv::xfeatures2d::SIFT::create(0,5,0.04,10,1.6);
            ROS_INFO("Keypoint detector: SIFT");
            return keyPointDetector;
        case STAR:
            keyPointDetector = cv::xfeatures2d::StarDetector::create(45,30,10,8,5);
            ROS_INFO("Keypoint detector: STAR");
            return keyPointDetector;
        case BRISK:
            keyPointDetector = cv::BRISK::create(30,3,1.0f);
            ROS_INFO("Keypoint detector: BRISK");
            return keyPointDetector;
        case ORB:
            keyPointDetector = cv::ORB::create(1000,1.2f,8,31,0,2,cv::ORB::FAST_SCORE,31,20);
            ROS_INFO("Keypoint detector: ORB");
            return keyPointDetector;
        case AKAZE:
            keyPointDetector = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,4,4,cv::KAZE::DIFF_PM_G2);
            ROS_INFO("Keypoint detector: AKAZE");
            return keyPointDetector;
        case FAST:
            keyPointDetector = cv::FastFeatureDetector::create(10,cv::FastFeatureDetector::TYPE_9_16);
            ROS_INFO("Keypoint detector: FAST");
            return keyPointDetector;
        default:
            keyPointDetector = cv::xfeatures2d::SURF::create(1000);
            ROS_ERROR("Could not find chosen keypoint detector. Choosing default: SURF");
            return keyPointDetector;
    }
}

cv::Ptr<cv::Feature2D> getDescriptors(int descriptorType, bool &binary) {
    cv::Ptr<cv::Feature2D> descriptorExtractor;
    switch(descriptorType) {
        case SURF:
            descriptorExtractor = cv::xfeatures2d::SURF::create(1000,4,5,false,false);
            binary = false;
            ROS_INFO("Descriptor: SURF");
            return descriptorExtractor;
        case SIFT:
            descriptorExtractor = cv::xfeatures2d::SIFT::create(0,5,0.04,10,1.6);
            binary = false;
            ROS_INFO("Descriptor: SIFT");
            return descriptorExtractor;
        case FREAK:
            descriptorExtractor = cv::xfeatures2d::FREAK::create(true,true,22.0f,4);
            binary = true;
            ROS_INFO("Descriptor: FREAK");
            return descriptorExtractor;
        case BRISK:
            descriptorExtractor = cv::BRISK::create(30,3,1.0f);
            binary = true;
            ROS_INFO("Descriptor: BRISK");
            return descriptorExtractor;
        case ORB:
            descriptorExtractor = cv::ORB::create(1000,1.2f,8,31,0,2,cv::ORB::FAST_SCORE,31,20);
            binary = true;
            ROS_INFO("Descriptor: ORB");
            return descriptorExtractor;
        case AKAZE:
            descriptorExtractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,4,4,cv::KAZE::DIFF_PM_G2);
            binary = true;
            ROS_INFO("Descriptor: AKAZE");
            return descriptorExtractor;
        case BRIEF:
            descriptorExtractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32,true);
            binary = true;
            ROS_INFO("Descriptor: BRIEF");
            return descriptorExtractor;
        default:
            descriptorExtractor = cv::xfeatures2d::SURF::create(1000);
            binary = false;
            ROS_INFO("Descriptor: SURF");
            return descriptorExtractor;
    }
}

cv::Ptr<cv::Feature2D> SetKeyPointsDetector(std::string typeKeyPoint) {
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

cv::Ptr<cv::Feature2D> SetDescriptorsExtractor(std::string typeDescriptor, bool &binary) {
    cv::Ptr<cv::Feature2D> extractor;
    if (typeDescriptor == "SURF") {
        extractor = cv::xfeatures2d::SURF::create(1000,4,5,false,false);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = false;
    } else if (typeDescriptor == "SIFT") {
        extractor = cv::xfeatures2d::SIFT::create(0,5,0.04,10,1.6);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = false;
    } else if (typeDescriptor == "BRISK") {
        extractor = cv::BRISK::create(30,3,1.0f);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = true;
    } else if (typeDescriptor == "FREAK") {
        extractor = cv::xfeatures2d::FREAK::create(true,true,22.0f,4);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = true;
    } else if (typeDescriptor == "ORB") {
        extractor = cv::ORB::create(1000,1.2f,8,31,0,2,cv::ORB::FAST_SCORE,31,20); // WTA_K = 3-4 -> HAMMING2
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = true;
    } else if (typeDescriptor == "AKAZE") {
        extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB,0,3,0.001f,4,4,cv::KAZE::DIFF_PM_G2);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = true;
    } else if (typeDescriptor == "BRIEF") {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(32, true);
        ROS_INFO("Descriptor: %s", typeDescriptor.c_str());
        binary = true;
    } else {
        ROS_ERROR("Could not find keypoint detector: %s\n\tChoosing default descriptor: SURF", typeDescriptor.c_str());
        extractor = cv::xfeatures2d::SURF::create(1000);
        binary = false;
    }
    return extractor;
}

std::vector<cv::DMatch> symmetryTest(const std::vector<cv::DMatch> &matches1, const std::vector<cv::DMatch> &matches2) {
    std::vector<cv::DMatch> symMatches;
    for (std::vector<cv::DMatch>::const_iterator matchIterator1 = matches1.begin();
         matchIterator1 != matches1.end(); ++matchIterator1) {
        for (std::vector<cv::DMatch>::const_iterator matchIterator2 = matches2.begin();
             matchIterator2 != matches2.end(); ++matchIterator2) {
            if ((*matchIterator1).queryIdx == (*matchIterator2).trainIdx &&
                (*matchIterator2).queryIdx == (*matchIterator1).trainIdx) {
                symMatches.push_back(
                        cv::DMatch((*matchIterator1).queryIdx, (*matchIterator1).trainIdx, (*matchIterator1).distance));
                break;
            }
        }
    }
    return symMatches;
}

CurrentMatch visualizeMatch(cv::Mat searchImage, cv::Mat objectImage, std::vector<cv::KeyPoint> keypointsObject,
                            std::vector<cv::KeyPoint> keypointsScene, std::vector<cv::DMatch> good_matches,
                            bool showKeypoints, int homographyType) {

    cv::Mat image_matches;

    if (showKeypoints) {
        cv::drawKeypoints(searchImage, keypointsScene, image_matches, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
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

bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r) {
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

int innerAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c) {
    cv::Point2f ab(b.x - a.x, b.y - a.y);
    cv::Point2f cb(b.x - c.x, b.y - c.y);

    double dot = (ab.x * cb.x + ab.y * cb.y); // dot product
    double cross = (ab.x * cb.y - ab.y * cb.x); // cross product

    double alpha = atan2(cross, dot);

    int angle = (int) floor(alpha * 180. / PI + 0.5);
    //alpha = alpha * 180 / PI;

    return abs(angle);
}

bool checkObjectInnerAngles(std::vector<cv::Point2f> scorner, int min, int max) {
    bool out = false;
    int c0 = innerAngle(scorner[3], scorner[0], scorner[1]);
    int c1 = innerAngle(scorner[0], scorner[1], scorner[2]);
    int c2 = innerAngle(scorner[1], scorner[2], scorner[3]);
    int c3 = innerAngle(scorner[2], scorner[3], scorner[0]);

    if (c0 > min && c0 < max && c1 > min && c1 < max && c2 > min && c2 < max && c3 > min && c3 < max) out = true;

    return out;
}

// Mean position
cv::Point2f getObjectCentroidMean(std::vector<cv::Point2f> scene) {
    cv::Point2f cen(0, 0);
    for (size_t i = 0; i < scene.size(); i++) {
        cen.x += scene[i].x;// + object_image.cols;
        cen.y += scene[i].y;
    }
    cen.x /= scene.size();
    cen.y /= scene.size();
    return cen;
}

// bbox
cv::Point2f getObjectCentroidBbox(std::vector<cv::Point2f> scene) {
    cv::Point2f pmin(1000000, 1000000);
    cv::Point2f pmax(0, 0);
    for (size_t i = 0; i < scene.size(); i++) {
        if (scene[i].x < pmin.x) pmin.x = scene[i].x;
        if (scene[i].y < pmin.y) pmin.y = scene[i].y;
        if (scene[i].x > pmax.x) pmax.x = scene[i].x;
        if (scene[i].y > pmax.y) pmax.y = scene[i].y;
    }
    cv::Point2f cen((pmax.x - pmin.x) / 2, (pmax.y - pmin.y) / 2);
    return cen;
}

double getXoffset(cv::Mat frame, std::vector<cv::Point2f> scorner) {
    cv::Point2f cen;
    double xOffset = 0.0;

    if (intersection(scorner[0], scorner[2], scorner[1], scorner[3], cen)) {
        xOffset = cen.x - frame.cols / 2;
    }

    return xOffset;
}

double getYoffset(cv::Mat frame, std::vector<cv::Point2f> scorner) {
    cv::Point2f cen;
    double yOffset = 0.0;

    if (intersection(scorner[0], scorner[2], scorner[1], scorner[3], cen)) {
        yOffset = cen.y - frame.rows / 2;
    }

    return yOffset;
}

double getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner) {

    double centerX = frame.cols / 2;
    double diffX = centerX - scorner[1].x;
    double x = (centerX - diffX) - scorner[0].x;

    double y = scorner[0].y - scorner[1].y;

    double angle = atan2(y, x) * 180 / PI;
    return angle;
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

int main(int argc, char **argv) {
    ros::init(argc, argv, "object_detection");
    ros::NodeHandle n;
    ros::Publisher pub1 = n.advertise<geometry_msgs::Pose2D>("/object_detection/offset1", 1);
    ros::Publisher pub2 = n.advertise<geometry_msgs::Pose2D>("/object_detection/offset2", 1);
    ros::Rate loop_rate(loop_frequency);

    // Read the image we are looking for
    object_image = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    object_image2 = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);
    if (!object_image.data || !object_image2.data) {
        ROS_ERROR(" --(!) Error reading images");
        return 0;
    }
    ROS_INFO("Loaded reference images:\n\t1: %s\n\t2: %s", argv[1], argv[2]);

    // Close if no camera could be opened
    if (!capture.open(0)) {
        ROS_ERROR(" --(!) Could not reach camera");
        return 0;
    }
    // Set resolution
    capture.set(CV_CAP_PROP_FRAME_WIDTH, resWidth);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, resHeight);
    ROS_INFO("Camera resolution: width=%f, height=%f",
             capture.get(CV_CAP_PROP_FRAME_WIDTH), capture.get(CV_CAP_PROP_FRAME_HEIGHT));

    // Detect keypoints and compute keypoints and descriptors for the reference image
    bool binary = false;
    bool bruteforce = true;
    cv::Ptr<cv::Feature2D> detector = SetKeyPointsDetector(DETECTOR_TYPE);
    cv::Ptr<cv::Feature2D> extractor = SetDescriptorsExtractor(EXTRACTOR_TYPE, binary);
//    cv::Ptr<cv::Feature2D> detector = getKeypoints(BRISK);
//    cv::Ptr<cv::Feature2D> extractor = getDescriptors(BRISK, binary);
    ROS_INFO("Binary matching: %d", binary);
    ROS_INFO("Bruteforce matching: %d", bruteforce);

    detector->detect(object_image, ko, deso);
    detector->detect(object_image2, ko2, deso2);
    extractor->compute(object_image, ko, deso);
    extractor->compute(object_image2, ko2, deso2);

    // Output the reference keypoints we are looking for
    cv::drawKeypoints(object_image, ko, ref_keypoints1);
    cv::drawKeypoints(object_image2, ko2, ref_keypoints2);
    cv::imwrite(ref_path1, ref_keypoints1);
    cv::imwrite(ref_path2, ref_keypoints2);
    ROS_INFO("Reference keypoints written:\n\t1: %s\n\t2: %s", ref_path1.c_str(), ref_path2.c_str());

    // Window to show the frames in
    cv::namedWindow(OPENCV_WINDOW, CV_WINDOW_FULLSCREEN);

    // Set up services
    ros::ServiceServer service1 = n.advertiseService("setProcessRunning", setProcessRunningCallBack);
    ros::ServiceServer service2 = n.advertiseService("getProcessRunning", getProcessRunningCallBack);

    cameraMatrix = getCameraMatrix(CALIBRATION_FILE);
    distCoeffs = getDistortionCoeff(CALIBRATION_FILE);

    // Loop until program is stopped
    ROS_INFO("Processing video stream at loop rate [Hz]: %f", loop_frequency);
    int count = 0;
    while (ros::ok()) {
        // Capture video frame
        videoFrame = captureFrame(false, false, capture);
        if (videoFrame.empty()) break; // || cv::waitKey(30) >= 0
        cv::waitKey(30);

        // Process frames
        if (running) {
            // Detect keypoints and descriptors
            detector->detect(videoFrame, ks);
            extractor->compute(videoFrame, ks, dess);

            // Match descriptors of reference and video frame
            std::vector<cv::DMatch> good_matches, good_matches2;
            if (!binary) {
                if (bruteforce) {
                    good_matches = bruteForce(deso, dess, cv::NORM_L1);
                    good_matches2 = bruteForce(deso2, dess, cv::NORM_L1);
                } else {
                    good_matches = knnMatchDescriptors(deso, dess, 0.9f);
                    good_matches2 = knnMatchDescriptors(deso2, dess, 0.9f);
                }
            } else {
                if(bruteforce) {
                    good_matches = bruteForce(deso, dess, cv::NORM_HAMMING);
                    good_matches2 = bruteForce(deso2, dess, cv::NORM_HAMMING);
                } else {
                    good_matches = knnMatchDescriptorsLSH(deso, dess, 0.9f);
                    good_matches2 = knnMatchDescriptorsLSH(deso2, dess, 0.9f);
                }
            }

            // Visualize matching
            if ((!ko.size() == 0 && !ks.size() == 0) && good_matches.size() >= 0) {

                match1 = visualizeMatch(videoFrame, object_image, ko, ks, good_matches, true, CV_RANSAC);
                match2 = visualizeMatch(match1.outFrame, object_image2, ko2, ks, good_matches2, false, CV_RANSAC);

                cv::imshow(OPENCV_WINDOW, match2.outFrame);
            } else {
                cv::imshow(OPENCV_WINDOW, videoFrame);
            }
        } else {
            cv::imshow(OPENCV_WINDOW, videoFrame);
        }


        if (count == 40) {
            // Do x, y, theta calculation
            running = true;
            //count = 0;
        }

        // ROS
        if (match2.sceneCorners.size() == 4 && checkObjectInnerAngles(match2.sceneCorners, 60, 120)) {
            pose_msg.theta = getObjectAngle(videoFrame, match2.sceneCorners);
            pose_msg.x = getXoffset(videoFrame, match2.sceneCorners);
            pose_msg.y = getYoffset(videoFrame, match2.sceneCorners);
            pub1.publish(pose_msg);
        }
        if (match1.sceneCorners.size() == 4 && checkObjectInnerAngles(match1.sceneCorners, 60, 120)) {
            pose_msg.theta = getObjectAngle(videoFrame, match1.sceneCorners);
            pose_msg.x = getXoffset(videoFrame, match1.sceneCorners);
            pose_msg.y = getYoffset(videoFrame, match1.sceneCorners);
            pub2.publish(pose_msg);
        }
        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    cv::destroyWindow(OPENCV_WINDOW);
    ROS_INFO("Object detection shutting down");

    return 0;
}
