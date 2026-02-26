#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>
#include "opencv2/imgcodecs.hpp"
#include <string>
#include <map>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>

// Dictionary that maps the string input to the actual dictionary
static int dictFromString(const std::string& name){
    static const std::map<std::string,int> m = {
        {"DICT_4X4_50", cv::aruco::DICT_4X4_50},
        {"DICT_4X4_100", cv::aruco::DICT_4X4_100},
        {"DICT_4X4_250", cv::aruco::DICT_4X4_250},
        {"DICT_4X4_1000", cv::aruco::DICT_4X4_1000},

        {"DICT_5X5_50", cv::aruco::DICT_5X5_50},
        {"DICT_5X5_100", cv::aruco::DICT_5X5_100},
        {"DICT_5X5_250", cv::aruco::DICT_5X5_250},
        {"DICT_5X5_1000", cv::aruco::DICT_5X5_1000},

        {"DICT_6X6_50", cv::aruco::DICT_6X6_50},
        {"DICT_6X6_100", cv::aruco::DICT_6X6_100},
        {"DICT_6X6_250", cv::aruco::DICT_6X6_250},
        {"DICT_6X6_1000", cv::aruco::DICT_6X6_1000},

        {"DICT_7X7_50", cv::aruco::DICT_7X7_50},
        {"DICT_7X7_100", cv::aruco::DICT_7X7_100},
        {"DICT_7X7_250", cv::aruco::DICT_7X7_250},
        {"DICT_7X7_1000", cv::aruco::DICT_7X7_1000},

        {"DICT_ARUCO_ORIGINAL", cv::aruco::DICT_ARUCO_ORIGINAL}
    };

    auto it = m.find(name);
    if (it == m.end())
        throw std::runtime_error("Unknown dictionary: " + name);

    return it->second;
}

bool readCameraParamsFromCommandLine(
    cv::Mat& camMatrix,
    cv::Mat& distCoeffs){

    // Extract the camera parameters file name 
    std::string filename = "/home/juan/Documents/IFROS/perception/LAB1Git/Lab1_Perception/src/calibration_v3"; 

    if (filename.empty()) {
        std::cerr << "Camera parameters file not provided." << std::endl;
        return false;
    }

    cv::FileStorage fs(filename, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Cannot open camera file: " << filename << std::endl;
        return false;
    }

    // Save the parameters from the file
    fs["camera_matrix"] >> camMatrix;
    fs["dist_coeffs"] >> distCoeffs;

    if (camMatrix.empty() || distCoeffs.empty()) {
        std::cerr << "Invalid camera parameters in file." << std::endl;
        return false;
    }

    // Return true if success
    return true;
}

// obtain from https://github.com/fdcl-gwu/aruco-markers/blob/master/draw_cube/src/main.cpp
void drawCubeWireframe(
    cv::InputOutputArray image, cv::InputArray camera_matrix,
    cv::InputArray dist_coeffs, cv::InputArray rvec, cv::InputArray tvec,
    float l
)
{
    float half_l = l / 2.0;

    // Project cube points
    std::vector<cv::Point3f> axis_points;
    axis_points.push_back(cv::Point3f(half_l, half_l, l));
    axis_points.push_back(cv::Point3f(half_l, -half_l, l));
    axis_points.push_back(cv::Point3f(-half_l, -half_l, l));
    axis_points.push_back(cv::Point3f(-half_l, half_l, l));
    axis_points.push_back(cv::Point3f(half_l, half_l, 0));
    axis_points.push_back(cv::Point3f(half_l, -half_l, 0));
    axis_points.push_back(cv::Point3f(-half_l, -half_l, 0));
    axis_points.push_back(cv::Point3f(-half_l, half_l, 0));

    std::vector<cv::Point2f> image_points;
    projectPoints(
        axis_points, rvec, tvec, camera_matrix, dist_coeffs, image_points
    );

    // Draw cube edges lines
    cv::line(image, image_points[0], image_points[1], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[0], image_points[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[0], image_points[4], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[1], image_points[2], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[1], image_points[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[2], image_points[3], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[2], image_points[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[3], image_points[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[4], image_points[5], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[4], image_points[7], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[5], image_points[6], cv::Scalar(255, 0, 0), 3);
    cv::line(image, image_points[6], image_points[7], cv::Scalar(255, 0, 0), 3);
}

int main(int argc, char *argv[])
{
    if (argc < 4)
    {
        std::cout << "Usage: ./calibration <video_source_no> <dictionary_name> <if_capture>" << std::endl;
        return -1;
    }

    // std::cout << std::atoi(argv[1]) << std::endl;
    cv::VideoCapture webCam(std::atoi(argv[1])); // VideoCapture object declaration. Usually 0 is the integrated, 2 is the first external USB one↪→

    if (webCam.isOpened() == false)
    { // Check if the VideoCapture object has been correctly associated to the webcam↪→
        std::cerr << "error: Webcam could not be connected." << std::endl;
        return -1;
    }
    // Extract parameters
    std::string dictionaryName = argv[2];
    std::string MarkerId1 = argv[3];
    std::string MarkerId2 = argv[4];
    float markerSize = std::stof(argv[5]);

    cv::Mat imgOriginal;  // input image

    static int dictionary_name = dictFromString(dictionaryName);

    // create the parametes for the detector and get the dictionary to use.
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);

    char charCheckForESCKey{0};

    cv::namedWindow("imgOriginal");

    // set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerSize/2.f, markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerSize/2.f, markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerSize/2.f, -markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerSize/2.f, -markerSize/2.f, 0);

    // create the matrix to store the marker corners and ids
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;


    cv::Mat camMatrix, distCoeffs;
    readCameraParamsFromCommandLine(camMatrix, distCoeffs);

    while (charCheckForESCKey != 27 && webCam.isOpened())
    {                                                 // loop until ESC key is pressed or webcam is lost
        bool frameSuccess = webCam.read(imgOriginal); // get next frame from input stream

        if (!frameSuccess || imgOriginal.empty())
        { // if the frame was not read or read wrongly
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        
        // detect the markers on the image
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        // cone the original image
        cv::Mat outputImage = imgOriginal.clone();
        // get the size of the markers detected
        size_t nMarkers = markerCorners.size();
        // create the vectors to store the rotation and translation vectors of each marker
        std::vector<cv::Vec3d> rvecs(2), tvecs(2);
        // create matrix to store the marker corners and ids
        std::vector<int> IDselected;
        std::vector<std::vector<cv::Point2f>> markerCornersselected, rejectedCandidatesselected;
        // Count the occurrences of the target value in the
        // vector
        int counter = 0;
        // check if the markers with the specified ids are detected
        int cnt1 = count(markerIds.begin(), markerIds.end(), std::stoi(MarkerId1));
        int cnt2 = count(markerIds.begin(), markerIds.end(), std::stoi(MarkerId2));
        if(!markerIds.empty() && cnt1 > 0 && cnt2 > 0) {
            // Calculate pose for each marker
            for (size_t i = 0; i < nMarkers; i++) {
                if(markerIds[i] == std::stoi(MarkerId1) || markerIds[i] == std::stoi(MarkerId2)) {
                    // sabe the corners and ids of the markers detected with the specified ids
                    IDselected.push_back(markerIds[i]);
                    markerCornersselected.push_back(markerCorners[i]);
                    rejectedCandidatesselected.push_back(rejectedCandidates[i]);

                    // Estimate the pose of the marker
                    cv::solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(counter), tvecs.at(counter));
                    // Draw the axis on the marker
                    cv::drawFrameAxes(outputImage, camMatrix, distCoeffs, rvecs[counter], tvecs[counter], markerSize * 1.5f, 2);
                    counter++;

                    // check if the markers with the specified ids are detected and draw them
                    int cnt1 = count(IDselected.begin(), IDselected.end(), std::stoi(MarkerId1));
                    int cnt2 = count(IDselected.begin(), IDselected.end(), std::stoi(MarkerId2));
                    if (cnt1 > 0 && cnt2 >0)
                    {
                    // draw the marker
                    cv::aruco::drawDetectedMarkers(outputImage, markerCornersselected, IDselected);
                    break;
                    }
                    
                    
                }
            }
        }
        // Calculate the relative translation between the two markers
        double delta_x = tvecs[1][0] - tvecs[0][0];
        double delta_y = tvecs[1][1] - tvecs[0][1];
        double delta_z = tvecs[1][2] - tvecs[0][2];

        // Adjust if the order is different
        if (delta_x < 0) { // marker 1 in at the right
            delta_x = -delta_x;
            delta_y = -delta_y;
            delta_z = -delta_z;

        }

        // Assign the text to show and format it
        char bufX[10], bufY[10], bufZ[10];
        std::snprintf(bufX, sizeof(bufX), "X = %.5f", delta_x);
        std::snprintf(bufY, sizeof(bufY), "Y = %.5f", delta_y);
        std::snprintf(bufZ, sizeof(bufZ), "Z = %.5f", delta_z);
        
        // Show the text in the image
        cv::putText(outputImage, bufX, cv::Point(20,40),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        
        cv::putText(outputImage, bufY, cv::Point(20,70),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);
        
        cv::putText(outputImage, bufZ, cv::Point(20,100),
        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);

        // Show output video results windows
        cv::imshow("imgOriginal", outputImage);
        charCheckForESCKey = cv::waitKey(1); // gets the key pressed
        if (charCheckForESCKey == 27) // ESC key pressed
        {
            break;
        }
    }
    return 0;
}