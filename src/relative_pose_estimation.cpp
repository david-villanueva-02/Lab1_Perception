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

bool readCameraParams(
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
    if (argc != 6)
    {   
        std::cout << "Usage is ./relative_pose_estimation <camera_id > <dictionary> <id1> <id2> <size>" << std::endl;
        return -1;
    }

    // Extract parameters
    std::string dictionaryName = argv[2];
    int markerId1 = std::stoi(argv[3]);
    int markerId2 = std::stoi(argv[4]);
    double markerSize = std::stod(argv[5]);
    
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
    const bool success = readCameraParams(camMatrix, distCoeffs);
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

        bool frameSuccess = webCam.read(imgOriginal); // get next frame from input stream

        if (!frameSuccess || imgOriginal.empty())
        { // if the frame was not read or read wrongly
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }
        
        // Show ArUcos in image
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        // std::vector<std::vector<cv::Point2f>> interest_markerCorners(2);


        // detect the markers on the image and draw the markers detected
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        cv::Mat outputImage = imgOriginal.clone();
        size_t nMarkers = markerIds.size();

        std::vector<cv::Vec3d> rvecs(2), tvecs(2);
        int counter = 0;

        if (!markerIds.empty()) {
            // Calculate pose for each marker
            for (size_t i = 0; i < nMarkers; ++i) {
                if(markerIds[i] == markerId1 || markerIds[i] == markerId2) {

                    // Store the marker corners of interest
                    // interest_markerCorners.push_back(markerCorners[i]);
                
                    // Estimate pose of the marker
                    cv::solvePnP(objPoints, markerCorners.at(i), camMatrix, distCoeffs, rvecs.at(counter), tvecs.at(counter), false, cv::SOLVEPNP_IPPE_SQUARE);

                    // Draw the axis for the marker
                    cv::drawFrameAxes(outputImage, camMatrix, distCoeffs, rvecs[counter], tvecs[counter], markerSize * 1.5f, 2);
                    
                    // Add a border to the marker detected
                    std::vector<std::vector<cv::Point2f>> oneCorners{ markerCorners[i] };
                    std::vector<int> oneIds{ markerIds[i] };
                    cv::aruco::drawDetectedMarkers(outputImage, oneCorners, oneIds);

                    counter++;

                    // Avoid unnecessary calculations 
                    if (counter == 2) { break; }
                }
            }

            // If all markers have been detected, we can calculate the relative pose and draw the line between the two markers
            if (counter == 2){
                
                // Transforms rotation vectors into rotation matrices
                cv::Mat R1, R2;
                cv::Rodrigues(rvecs[0], R1);
                cv::Rodrigues(rvecs[1], R2);
                
                // Get the translation vectors of the two markersstd::
                cv::Mat t1 = (cv::Mat_<double>(3,1) << tvecs[0][0], tvecs[0][1], tvecs[0][2]);
                cv::Mat t2 = (cv::Mat_<double>(3,1) << tvecs[1][0], tvecs[1][1], tvecs[1][2]);
                
                // Deinfe the 3D points relative to the markers' coordinate systems
                std::vector<cv::Point3f> Pm1 = { cv::Point3f(0.f, 0.f, 0.f) }; // marker-frame center for marker 1
                std::vector<cv::Point3f> Pm2 = { cv::Point3f(0.f, 0.f, 0.f) }; // marker-frame center for marker 2
                std::vector<cv::Point2f> pim1, pim2; // projected marker centers into the image plane
                
                // Project the marker centers into the image plane
                cv::projectPoints(Pm1, rvecs[0], tvecs[0], camMatrix, distCoeffs, pim1);
                cv::projectPoints(Pm2, rvecs[1], tvecs[1], camMatrix, distCoeffs, pim2);
                
                // U coordinates of both centers
                double u1 = pim1[0].x;
                double u2 = pim2[0].x;

                // translation of marker 2 expressed in marker 1 coordinates
                cv::Mat t_rel;
                if (u2 > u1) {
                    t_rel = R1.t() * (t2 - t1);
                }
                else {
                    t_rel = R2.t() * (t1 - t2);
                }
                
                // Get relative translation in x, y and z
                double delta_x = t_rel.at<double>(0);
                double delta_y = t_rel.at<double>(1);
                double delta_z = t_rel.at<double>(2);

                // cv::Rodrigues

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
            }
        }

        cv::imshow("pose_estimation", outputImage);
        int key = cv::waitKey(1);
    }

    return 0;
}