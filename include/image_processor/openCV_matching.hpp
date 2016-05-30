//
// Original author: Asgeir Bjoerkedal. Created: 10.03.16. Last edit: 30.05.16.
//
// The class implements methods from OpenCV and is designed for use in an object detection application.
// It encompasses capturing of video frames, processing video frames by numerous keypoint
// detectors and descriptor extractors, matching algorithms, visualization and computation
// of object image coordinates and orientation.
//
// Created as part of the software solution for a Master's thesis in Production Technology at NTNU Trondheim.
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

namespace robotcam {

    struct CurrentMatch {
        /*! The frame with visualized keypoints and matching. */
        cv::Mat outFrame;
        /*! The corners of the matched object in the scene. */
        std::vector<cv::Point2f> sceneCorners;
    };

    class OpenCVMatching {
    public:
        /*!
         * \brief Get a camera matrix from XML or YAML file.
         * \param path The path of the file.
         * \return The camera matrix.
         */
        cv::Mat getCameraMatrix(const std::string path);

        /*!
         * \brief Get the distortion coefficients from XML or YAML file.
         * \param path The path of the file.
         * \return The distortion coefficients.
         */
        cv::Mat getDistortionCoeff(const std::string path);

        /*!
         * \brief Check the actual type openCV cv::Mat.
         * \param type The type of a matrix.
         * \return The matrix type as string.
         */
        std::string type2str(int type);

        /*!
         * \brief Capture a frame from a connected web camera.
         * \param color True for RGB. False for grayscale.
         * \param undistort True for correction for lens distortion. False for no correction.
         * \param capture The object capturing a frame from the web camera.
         * \param cameraMatrix The camera matrix (K-matrix) of the web camera.
         * \param distCoeffs The distortion coefficients of the web camera.
         * \return The current video frame.
         *
         * Capture a frame in color/grayscale and with or without lens distortion.
         */
        cv::Mat captureFrame(bool color, bool undistort, cv::VideoCapture capture, cv::Mat cameraMatrix, cv::Mat distCoeffs);

        /*!
         * \brief Capture a frame from a connected web camera.
         * \param color The boolean determining RGB or grayscale video frame.
         * \param capture The object capturing the video stream from the camera.
         * \return The current video frame.
         *
         * Capture either with color or grayscale.
         */
        cv::Mat captureFrame(bool color, cv::VideoCapture capture);

        /*!
         * \brief Flann based nearest neighbour matching.
         * \param descriptors_object The descriptors of the query image.
         * \param descriptors_scene The descriptors of the training scene image.
         * \param nnratio The nearest neighbour ratio for distance filtering.
         * \return The good matches.
         */
        std::vector<cv::DMatch> knnMatchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nnratio);

        /*!
         * \brief Flann based nearest neighbour with LSH index for binary matching.
         * \param descriptors_object The descriptors of the query image.
         * \param descriptors_scene The descriptors of the training scene image.
         * \param nndrRatio The nearest neighbour ratio for distance filtering.
         * \return The good matches.
         */
        std::vector<cv::DMatch> knnMatchDescriptorsLSH(cv::Mat descriptors_object, cv::Mat descriptors_scene, float nndrRatio);

        /*!
         * \brief Flann based matching.
         * \param descriptors_object The descriptors of the query image.
         * \param descriptors_scene The descriptors of the training scene image.
         * \return The good matches.
         */
        std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene);

        /*!
         * \brief Bruteforce nearest neighbour matching.
         * \param descriptors_object The descriptors of the query image.
         * \param descriptors_scene The descriptors of the training scene image.
         * \param normType The distance type, e.g. NORM_L1, NORM_L2, NORM_HAMMING.
         */
        std::vector<cv::DMatch> bruteForce(cv::Mat descriptors_object, cv::Mat descriptors_scene, int normType);

        /*!
         * \brief Set a keypoint detector based on a input string.
         * \param typeKeyPoint The input string as an acronym for wanted algorithm, e.g. SIFT, SURF.
         * \return The keypoint detector.
         */
        cv::Ptr<cv::Feature2D> setKeyPointsDetector(std::string typeKeyPoint);

        /*!
         * \brief Set a descriptor extractor based on a input string.
         * \param typeDescriptor The input string as an acronym for wanted algorithm, e.g. SIFT, SURF.
         * \param binary Reference to a matching control boolean. True if real-valued descriptor, False if binary.
         * \return The descriptor extractor.
         */
        cv::Ptr<cv::Feature2D> setDescriptorsExtractor(std::string typeDescriptor, bool &binary);

        /*!
         * \brief Visualize a object matching using homography.
         * \param searchImage The training scene image.
         * \param objectImage The query image.
         * \param keypointsObject The keypoints of the query image.
         * \param keypointsScene The keypoints of the training scene image.
         * \param good_matches The good matches between query and training image.
         * \param showKeypoints True for visualized keypoints. False for no drawn keypoints.
         * \param homographyType The homography type, e.g. CV_RANSAC or CV_LMEDS.
         * \return The current match holding an image with visualized matching and the object corners in training scene.
         */
        CurrentMatch visualizedMatch(cv::Mat searchImage, cv::Mat objectImage, std::vector<cv::KeyPoint> keypointsObject, std::vector<cv::KeyPoint> keypointsScene, std::vector<cv::DMatch> good_matches, bool showKeypoints, int homographyType);

        /*!
         * \brief Check if the inner angles of a square or rectangle is within min and max angle.
         * \param scorner The training scene corners of the matched object.
         * \param min The minimum angle in degrees.
         * \param max The maximum angle in degrees.
         * \return True if angle is within min and max. False otherwise.
         */
        bool checkObjectInnerAngles(std::vector<cv::Point2f> scorner, int min, int max);

        /*!
         * \brief Get the pixel offset in x-direction of the matched object center related to the image frame center.
         * \param frame The training scene image.
         * \param scorner The scene corners of the matched object.
         * \return The object offset in x-direction.
         */
        double getXoffset(cv::Mat frame, std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the pixel offset in y-direction of the matched object center related to the image frame center.
         * \param frame The training scene image.
         * \param scorner The scene corners of the matched object.
         * \return The object pixel offset in y-direction.
         */
        double getYoffset(cv::Mat frame, std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the pixel coordinate x of the matched object center.
         * \param scorner The scene corners of the matched object.
         * \return The pixel coordinate x.
         */
        double getXpos(std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the pixel coordinate y of the matched object center.
         * \param scorner The scene corners of the matched object.
         * \return The pixel coordinate y.
         */
        double getYpos(std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the angle of in-plane rotation of the matched object.
         * \param frame The training scene image.
         * \param scorner The scene corners of the matched object.
         * \return The object angle in degrees.
         */
        double getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the normalized image coordinates of the matched object scaled with lambda.
         * \param x The pixel coordinate x.
         * \param y The pixel coordinate y.
         * \param lambda The depth to object along optical axis from camera lens.
         * \param camera_matrix The K-matrix of the camera.
         * \return The normalized image coordinates scaled with lambda.
         */
        Eigen::Vector3d getNormImageCoords(double x, double y, double lambda, cv::Mat camera_matrix);

    private:
        /*!
         * \brief Get the intersection point of two lines.
         * \param o1 The origin point of the first line.
         * \param p1 The end point of the first line.
         * \param o2 The origin point of the second line.
         * \param p2 The end point of the second line.
         * \param r The intersection point referenced.
         * \return The boolean whether an intersection was found. True if found. False otherwise.
         */
        bool intersection(cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2, cv::Point2f &r);

        /*!
         * \brief Get the inner angle using three points.
         * \param a The first point.
         * \param b The origin of the angle.
         * \param c The second point.
         * \return The angle in degrees.
         */
        int innerAngle(cv::Point2f a, cv::Point2f b, cv::Point2f c);
    };
}
#endif //IMAGE_PROCESSOR_OPENCV_MATCHING_HPP
