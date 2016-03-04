//
// Created by minions on 17.02.16.
//

#include "../include/image_processor/openCV_tester.hpp"

namespace robotCam {
    // Constructor
    openCV_tester::openCV_tester() { }

    // Deconstructor
    openCV_tester::~openCV_tester() { }
}

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

std::vector<cv::DMatch> knnMatchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nndrRatio) {

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

        if (m1.distance <= nndrRatio * m2.distance)
            good_matches.push_back(m1);
    }

    return good_matches;
}

std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene) {

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

CurrentMatch visualizeMatch(cv::Mat searchImage, cv::Mat objectImage, cv::Point2f outCentroid,
                            std::vector<cv::KeyPoint> keypointsObject, std::vector<cv::KeyPoint> keypointsScene,
                            std::vector<cv::DMatch> good_matches, bool showMatches) {

    cv::Mat image_matches;

    if (!showMatches) {
        cv::drawKeypoints(searchImage, keypointsScene, image_matches, cv::Scalar::all(-1),
                          cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    } else {
        if (!keypointsObject.size() == 0 && !keypointsScene.size() == 0) {
            cv::drawMatches(objectImage, keypointsObject, searchImage, keypointsScene, good_matches, image_matches,
                            cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                            cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        }
    }

    std::vector<cv::Point2f> obj;
    std::vector<cv::Point2f> scene;

    for (size_t i = 0; i < good_matches.size(); i++) {
        // Retrieve the keypoints from good matches
        obj.push_back(keypointsObject[good_matches[i].queryIdx].pt);
        scene.push_back(keypointsScene[good_matches[i].trainIdx].pt);
    }

    // Perform Homography to find a perspective transformation between two planes.
    cv::Mat H = cv::findHomography(obj, scene, CV_RANSAC);

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

        if (!showMatches) {
            // Draw lines surrounding the object
            cv::line(image_matches, sceneCorners[0], sceneCorners[1], cv::Scalar(0, 255, 0), 2); //TOP line
            cv::line(image_matches, sceneCorners[1], sceneCorners[2], cv::Scalar(0, 255, 0), 2); //RIGHT line
            cv::line(image_matches, sceneCorners[2], sceneCorners[3], cv::Scalar(0, 255, 0), 2); //BOTTOM line
            cv::line(image_matches, sceneCorners[3], sceneCorners[0], cv::Scalar(0, 255, 0), 2); //LEFT line

            cv::line(image_matches, sceneCorners[0], sceneCorners[2], cv::Scalar(0, 255, 0), 1); //DIAGONAL 0-2
            cv::line(image_matches, sceneCorners[1], sceneCorners[3], cv::Scalar(0, 255, 0), 1); //DIAGONAL 1-3
        } else {
            // Draw lines with objectImage offset
            cv::line(image_matches, sceneCorners[0] + cv::Point2f(objectImage.cols, 0),
                     sceneCorners[1] + cv::Point2f(objectImage.cols, 0), cv::Scalar(0, 255, 0), 4);

            cv::line(image_matches, sceneCorners[1] + cv::Point2f(objectImage.cols, 0),
                     sceneCorners[2] + cv::Point2f(objectImage.cols, 0), cv::Scalar(0, 255, 0), 4);

            cv::line(image_matches, sceneCorners[2] + cv::Point2f(objectImage.cols, 0),
                     sceneCorners[3] + cv::Point2f(objectImage.cols, 0), cv::Scalar(0, 255, 0), 4);

            cv::line(image_matches, sceneCorners[3] + cv::Point2f(objectImage.cols, 0),
                     sceneCorners[0] + cv::Point2f(objectImage.cols, 0), cv::Scalar(0, 255, 0), 4);
        }
    }

    // Draw circles in center pixel of the video stream
    if (searchImage.rows > 60 && searchImage.cols > 60) {
        cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 5, CV_RGB(255, 0, 0));
        cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 10, CV_RGB(0, 255, 0));
        cv::circle(image_matches, cv::Point(searchImage.cols / 2, searchImage.rows / 2), 15, CV_RGB(0, 0, 255));
    }

    // Centroid
    if (intersection(sceneCorners[0], sceneCorners[2], sceneCorners[1], sceneCorners[3], outCentroid)) {
        cv::Point2f cen = outCentroid;
        if (!showMatches) {
            cv::circle(image_matches, cen, 10, cv::Scalar(0, 0, 255), 2);
        } else {
            cv::circle(image_matches, cen + cv::Point2f(objectImage.cols, 0), 10, cv::Scalar(0, 0, 255), 2);
        }
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

bool innerAngle(std::vector<cv::Point2f> scorner) {
    for(size_t i=0; i<scorner.size(); ++i) {
        double px1;
        double py1;
        double px2;
        double py2;
        double cx1 = scorner[i].x;
        double cy1 = scorner[i].y;

        double dist1 = sqrt((px1 - cx1) * (px1 - cx1) + (py1 - cy1) * (py1 - cy1));
        double dist2 = sqrt((px2 - cx1) * (px2 - cx1) + (py2 - cy1) * (py2 - cy1));

        double Ax, Ay;
        double Bx, By;
        double Cx, Cy;

        //find closest point to C
        //printf("dist = %lf %lf\n", dist1, dist2);

        Cx = cx1;
        Cy = cy1;
        if (dist1 < dist2) {
            Bx = px1;
            By = py1;
            Ax = px2;
            Ay = py2;
        } else {
            Bx = px2;
            By = py2;
            Ax = px1;
            Ay = py1;
        }

        double Q1 = Cx - Ax;
        double Q2 = Cy - Ay;
        double P1 = Bx - Ax;
        double P2 = By - Ay;

        double A = acos((P1 * Q1 + P2 * Q2) / (sqrt(P1 * P1 + P2 * P2) * sqrt(Q1 * Q1 + Q2 * Q2)));

        A = A * 180 / PI;
    }
    return true;
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
    //cen.x = scene_corners[0].x+(scene_corners[1].x-scene_corners[0].x)/2;
    cen.x = scorner[0].x + (scorner[2].x - scorner[0].x) / 2;
    double xOffset = cen.x - frame.cols / 2;
    return xOffset;
}

double getYoffset(cv::Mat frame, std::vector<cv::Point2f> scorner) {
    cv::Point2f cen;
    //cen.y = scene_corners[0].y+(scene_corners[3].y-scene_corners[0].y)/2;
    cen.y = scorner[0].y + (scorner[2].y - scorner[0].y) / 2;
    double yOffset = cen.y - frame.rows / 2;
    return yOffset;
}

double getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner) {
    //std::cout << scorner.at(2) << std::endl;
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
    ros::Publisher pub = n.advertise<geometry_msgs::Pose2D>("/object_detection/offset", 1);
    ros::Rate loop_rate(loop_frequency);

    bool showMatches = true;

    // Read the image we are looking for
    object_image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);
    object_image2 = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    if (!object_image.data) {
        ROS_ERROR(" --(!) Error reading image: %s", argv[1]);
        return 0; //-1
    }
    ROS_INFO("Loaded reference image: %s", argv[1]);
    if (!object_image2.data) {
        ROS_ERROR(" --(!) Error reading image: %s", argv[2]);
        return 0; //-1
    }
    ROS_INFO("Loaded reference image: %s", argv[2]);

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
    f2d->detectAndCompute(object_image, cv::Mat(), ko, deso);
    f2d->detectAndCompute(object_image2, cv::Mat(), ko2, deso2);


    // Output the reference keypoints we are looking for
    cv::drawKeypoints(object_image, ko, ref_keypoints);
    cv::imwrite(ref_path, ref_keypoints);
    ROS_INFO("Reference keypoints written to: \n\t%s", ref_path.c_str());

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
        videoFrame = captureFrame(true, true, capture);
        if (videoFrame.empty()) break; // || cv::waitKey(30) >= 0
        cv::waitKey(30);

        CurrentMatch match1;
        CurrentMatch match2;

        // Process frames
        if (running) {
            // Detect keypoints and descriptors
            f2d->detectAndCompute(videoFrame, cv::Mat(), ks, dess);

            // Match descriptors of reference and video frame
            std::vector<cv::DMatch> good_matches = knnMatchDescriptors(deso, dess, 0.9f);
            std::vector<cv::DMatch> good_matches2 = knnMatchDescriptors(deso2, dess, 0.9f);

            if ((!ko.size() == 0 && !ks.size() == 0) && good_matches.size() >= 7) {

                match1 = visualizeMatch(videoFrame, object_image, centroid1, ko, ks, good_matches, false);
                match2 = visualizeMatch(match1.outFrame, object_image2, centroid2, ko2, ks, good_matches2, false);

                cv::imshow(OPENCV_WINDOW, match2.outFrame);
            } else {
                cv::imshow(OPENCV_WINDOW, videoFrame);
            }
        } else {
            // Show the video frame
            cv::imshow(OPENCV_WINDOW, videoFrame);
        }


        if (count == 40) {
            std::cout << count << std::endl;

            // Do x, y, theta calculation
            running = true;
            //count = 0;
        }

        // ROS
        if (match2.sceneCorners.size() == 4) {
            pose_msg.theta = getObjectAngle(videoFrame, match2.sceneCorners);
            pose_msg.x = getXoffset(videoFrame, match2.sceneCorners);
            pose_msg.y = getYoffset(videoFrame, match2.sceneCorners);
            pub.publish(pose_msg);
        }
        ros::spinOnce();
        loop_rate.sleep();
        ++count;
    }
    cv::destroyWindow(OPENCV_WINDOW);
    ROS_INFO("Object detection shutting down");

    return 0;
}
