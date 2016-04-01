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
         * \param color The boolean determining RGB or grayscale video frame.
         * \param undistort The boolean determining if the output frame is undistorted using a K-matrix.
         * \param capture The object capturing a frame from the web camera.
         * \param cameraMatrix The camera matrix (K-matrix) of the web camera.
         * \param distCoeffs The distortion coefficients of the web camera.
         * \return The current video frame.
         *
         * Capture a frame in color/grayscale and with or without lens distortion.
         */
        cv::Mat captureFrame(bool color, bool undistort, cv::VideoCapture capture, cv::Mat cameraMatrix,
                             cv::Mat distCoeffs);

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
         * \param descriptors_object The descriptors of the object.
         * \param descriptors_scene The descriptors of the scene frame.
         * \param nnratio The nearest neighbour ratio for distance filtering.
         * \return The good matches.
         */
        std::vector<cv::DMatch> knnMatchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene,
                                                    float nnratio);

        /*!
         * \brief Flann based nearest neighbour with LSH index for binary matching.
         * \param descriptors_object The descriptors of the object.
         * \param descriptors_scene The descriptors of the scene frame.
         * \param nndrRatio The nearest neighbour ratio for distance filtering.
         * \return The good matches.
         */
        std::vector<cv::DMatch> knnMatchDescriptorsLSH(cv::Mat descriptors_object, cv::Mat descriptors_scene,
                                                       float nndrRatio);

        /*!
         * \brief Flann based matching.
         * \param descriptors_object The descriptors of the object.
         * \param descriptors_scene The descriptors of the scene frame.
         * \return The good matches.
         */
        std::vector<cv::DMatch> matchDescriptors(cv::Mat descriptors_object, cv::Mat descriptors_scene);

        /*!
         * \brief Bruteforce nearest neighbour matching.
         * \param descriptors_object The descriptors of the object.
         * \param descriptors_scene The descriptors of the scene frame.
         * \param normType The distance type, eg. NORM_L1, NORM_L2, NORM_HAMMING.
         */
        std::vector<cv::DMatch> bruteForce(cv::Mat descriptors_object, cv::Mat descriptors_scene, int normType);

        /*!
         * \brief Set a keypoint detector based on a input string.
         * \param typeKeyPoint The input string.
         * \return The keypoint detector.
         */
        cv::Ptr<cv::Feature2D> setKeyPointsDetector(std::string typeKeyPoint);

        /*!
         * \brief Set a descriptor extractor based on a input string.
         * \param typeDescriptor The input string.
         * \param binary The reference to a matching control boolean.
         * \return The descriptor extractor.
         */
        cv::Ptr<cv::Feature2D> setDescriptorsExtractor(std::string typeDescriptor, bool &binary);

        /*!
         * \brief Visualize a object matching using homography.
         * \param searchImage The scene frame.
         * \param objectImage The object image.
         * \param keypointsObject The keypoints of the object.
         * \param keypointsScene The keypoints of the scene.
         * \param good_matches The good matches between scene and object.
         * \param showKeypoints The boolean determining whether keypoints are drawn.
         * \param homographyType The homography type, eg. CV_RANSAC or CV_LMEDS.
         * \return The current match holding a image frame with visualized matching and the object corners in the scene.
         */
        CurrentMatch visualizedMatch(cv::Mat searchImage, cv::Mat objectImage,
                                     std::vector<cv::KeyPoint> keypointsObject,
                                     std::vector<cv::KeyPoint> keypointsScene, std::vector<cv::DMatch> good_matches,
                                     bool showKeypoints, int homographyType);

        /*!
         * \brief Check if the inner angles of a square or rectangle is within min and max.
         * \param scorner The scene corners of the matched object.
         * \param min The minimum angle in degrees.
         * \param max The maximum angle in degrees.
         * \return The boolean
         */
        bool checkObjectInnerAngles(std::vector<cv::Point2f> scorner, int min, int max);

        /*!
         * \brief Get the pixel offset in x-direction of the matched object center related to the image frame center.
         * \param frame The image frame.
         * \param scorner The scene corners of the matched object.
         * \return The object offset in x-direction.
         */
        double getXoffset(cv::Mat frame, std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the pixel offset in y-direction of the matched object center related to the image frame center.
         * \param frame The image frame.
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
         * \brief Get the angle of rotation of the matched object.
         * \param frame The image frame.
         * \param scorner The scene corners of the matched object.
         * \return The object angle in degrees.
         */
        double getObjectAngle(cv::Mat frame, std::vector<cv::Point2f> scorner);

        /*!
         * \brief Get the normalized image coordinates of the matched object.
         * \param x The pixel coordinate x.
         * \param y The pixel coordinate y.
         * \param lambda The depth to object from camera lens.
         * \param camera_matrix The K-matrix of the camera.
         * \return The normalized image coordinates.
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
         * \return The boolean whether an intersection was found.
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
