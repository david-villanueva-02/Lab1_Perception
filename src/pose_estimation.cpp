#include "opencv2/core/core.hpp"
#include <opencv2/core.hpp>
#include <opencv2/core/persistence.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>

#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d.hpp>

#include <iostream>
#include <string>
#include <map>

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
    std::string filename = "/home/david/Documents/IFRoS/Perception/labs/lab1/src/calibration_v2.yaml"; 

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

int main(int argc, char *argv[]){

    // Check the number of arguments
    if (argc != 5)
    {   
        std::cout << "Usage is ./pose_estimation <camera_id > <dictionary> <id> <size>" << std::endl;
        return -1;
    }

    // Extract parameters
    std::string dictionaryName = argv[2];
    int markerId = std::stoi(argv[3]);
    int markerSize = std::stoi(argv[4]);
    
    // Process parameters
    static int dictionary_name = dictFromString(dictionaryName);
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);

    // Process parameters for the detector
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
    detectorParams.cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    detectorParams.cornerRefinementWinSize = 5;   
    detectorParams.cornerRefinementMaxIterations = 30;
    detectorParams.cornerRefinementMinAccuracy = 0.1;
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    // Read camera parameters
    cv::Mat camMatrix, distCoeffs;
    const bool success = readCameraParamsFromCommandLine(camMatrix, distCoeffs);
    if (!success) {
        std::cerr << "Failed to read camera parameters." << std::endl;
        return -1;
    }

    // Print the camera parameters
    std::cout << "Camera Matrix:\n" << camMatrix << std::endl;
    std::cout << "Distortion Coefficients:\n" << distCoeffs << std::endl;

    // set coordinate system
    cv::Mat objPoints(4, 1, CV_32FC3);
    objPoints.ptr<cv::Vec3f>(0)[0] = cv::Vec3f(-markerSize/2.f, markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[1] = cv::Vec3f(markerSize/2.f, markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[2] = cv::Vec3f(markerSize/2.f, -markerSize/2.f, 0);
    objPoints.ptr<cv::Vec3f>(0)[3] = cv::Vec3f(-markerSize/2.f, -markerSize/2.f, 0);
    
    // Open video capture
    cv::VideoCapture webCam(std::atoi(argv[1])); 
    if (!webCam.isOpened()) {
        std::cerr << "Could not open camera index " << std::atoi(argv[1]) << "\n";
        return 1;
    }

    // Get key pressed from the keyboard
    char charCheckForKey{0};
    cv::Mat imgOriginal;  // input image

    while (charCheckForKey != 27 && webCam.isOpened()){    
        charCheckForKey = cv::waitKey(1);
        bool frameSuccess = webCam.read(imgOriginal); // get next frame from input stream

        if (!frameSuccess || imgOriginal.empty())
        { // if the frame was not read or read wrongly
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }
        
        // Show ArUcos in image
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        // detect the markers on the image and draw the markers detected
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        cv::Mat outputImage = imgOriginal.clone();
        size_t nMarkers = markerIds.size();
        std::vector<cv::Vec3d> rvecs(nMarkers), tvecs(nMarkers);

        if (!markerIds.empty()) {
            // Calculate pose for each marker
            for (size_t i = 0; i < nMarkers; ++i) {
                if(markerIds[i] != markerId) { continue; }
                
                // Estimate pose of the marker
                cv::solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(i), tvecs.at(i));

                // Draw the axis for the marker
                cv::drawFrameAxes(outputImage, camMatrix, distCoeffs, rvecs[i], tvecs[i], markerSize * 1.5f, 2);
                
                // Add a border to the marker detected
                std::vector<std::vector<cv::Point2f>> oneCorners{ markerCorners[i] };
                std::vector<int> oneIds{ markerIds[i] };
                cv::aruco::drawDetectedMarkers(outputImage, oneCorners, oneIds);

                // Put coordinates in the image
                // XYZ text
                cv::Vec3d tvec = tvecs[i];
                char bufX[10], bufY[10], bufZ[10];
                std::snprintf(bufX, sizeof(bufX), "X = %.5f", tvec[0]);
                std::snprintf(bufY, sizeof(bufY), "Y = %.5f", tvec[1]);
                std::snprintf(bufZ, sizeof(bufZ), "Z = %.5f", tvec[2]);

                cv::putText(outputImage, bufX, cv::Point(20,40),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                
                cv::putText(outputImage, bufY, cv::Point(20,70),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);
                
                cv::putText(outputImage, bufZ, cv::Point(20,100),
                cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2, cv::LINE_AA);

                break;
            }
        }

        cv::imshow("pose_estimation", outputImage);
    }

    return 0;
}