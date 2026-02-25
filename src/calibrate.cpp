#include "opencv2/core/core.hpp"
#include <opencv2/core/mat.hpp>
// #include <opencv2/aruco.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include "opencv2/imgcodecs.hpp"

#include <iostream>
#include <string>
#include <map>
#include <algorithm>

// Dictionary that maps the string input to the actual dictionary
const int dictFromString(const std::string& name){
    const std::map<std::string,int> m = {
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

int main(int argc, char *argv[]){
    if (argc < 8){
        std::cerr << "Usage: ./calibrate <video_source_no> <dictionary_name> <detector_parameters> <board_rows> <board_cols> <marker_size> <separation> <file_name>" << std::endl;
        return -1;
    }

    // Extract parameters
    std::string dictionaryName = argv[2];
    std::string detectorParamsFile = argv[3];
    int boardRows = std::atoi(argv[4]);
    int boardCols = std::atoi(argv[5]);
    float markerSize = std::stof(argv[6]);
    float separation = std::stof(argv[7]);
    std::string fileName = argv[8]; 

    // Process parameters.
    static int dictionary_name = dictFromString(dictionaryName);
    cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);
    cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();

    cv::VideoCapture webCam(std::atoi(argv[1])); // VideoCapture object declaration. Usually 0 is the integrated, 2 is the first external USB one↪→

    if (webCam.isOpened() == false){
        // Check if the VideoCapture object has been correctly associated to the webcam↪→ 
        std::cerr << "error: Webcam could not be connected." << std::endl;
        return -1;
    }

    cv::Mat imgOriginal;  // input image
    int nframe = 0; // Frame counter

    char charCheckForKey{0};

    // Collected frames for calibration
    std::vector<std::vector<std::vector<cv::Point2f>>> allMarkerCorners;
    std::vector<std::vector<int>> allMarkerIds;
    cv::Size imageSize;

    // Loop to take images
    while (charCheckForKey != 27 && webCam.isOpened()){                                              
        bool frameSuccess = webCam.read(imgOriginal);

        if (!frameSuccess || imgOriginal.empty()){ 
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }
        charCheckForKey = cv::waitKey(1);
        
        // Show ArUcos in image
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;

        
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        // Detect the markers on the imageand draw the markers detected
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        cv::Mat outputImage = imgOriginal.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

        cv::imshow("imgOriginal", outputImage);

        // Capture image when the c key is pressed and save it
        if (charCheckForKey == 'c' && !markerIds.empty()){
            
            printf("Capturing image %d\n", nframe);
            std::string fileName = "/home/david/Documents/IFRoS/Perception/labs/lab1/images/image_" + std::to_string(nframe) + ".jpg";

            // Save data for calibration
            allMarkerCorners.push_back(markerCorners);
            allMarkerIds.push_back(markerIds);
            imageSize = imgOriginal.size();

            // cv::imwrite(fileName, imgOriginal);
            nframe++;
        }
        // Gets the key pressed
        charCheckForKey = cv::waitKey(1); 
    }

    // Calibration process
    cv::Mat cameraMatrix, distCoeffs;

    // Gridboard
    cv::aruco::GridBoard gridboard(cv::Size(boardRows, boardCols), markerSize, separation, dictionary);
    cv::aruco::ArucoDetector detector(dictionary, detectorParams);

    // if(calibrationFlags & CALIB_FIX_ASPECT_RATIO) {
    //     cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    //     cameraMatrix.at<double>(0, 0) = aspectRatio;
    // }

    // Prepare data for calibration
    std::vector<cv::Point3f> objectPoints;
    std::vector<cv::Point2f> imagePoints;
    std::vector<cv::Mat> processedObjectPoints, processedImagePoints;
    size_t nFrames = allMarkerCorners.size();
    
    for(size_t frame = 0; frame < nFrames; frame++) {
        cv::Mat currentImgPoints, currentObjPoints;

        gridboard.matchImagePoints(allMarkerCorners[frame], allMarkerIds[frame], currentObjPoints, currentImgPoints);

        if(currentImgPoints.total() > 0 && currentObjPoints.total() > 0) {
            processedImagePoints.push_back(currentImgPoints);
            processedObjectPoints.push_back(currentObjPoints);
        }
    }

    // Calibrate camera
    std::vector<cv::Mat> rvecs, tvecs;
    double repError = cv::calibrateCamera(processedObjectPoints, processedImagePoints, imageSize, cameraMatrix, distCoeffs,
                                    cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray(), cv::noArray()
                                    // calibrationFlags
                                );
    // Build markerCounterPerFrame
    std::vector<int> markerCounterPerFrame;
    markerCounterPerFrame.reserve(allMarkerIds.size());
    for (const auto& ids : allMarkerIds) {
        markerCounterPerFrame.push_back(static_cast<int>(ids.size()));
    }

    // cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(boardCols, boardRows, markerSize, separation, dictionary);
    // double repError = cv::aruco::calibrateCameraAruco(
    //     allMarkerCorners,          // per-frame corners
    //     allMarkerIds,              // per-frame ids
    //     markerCounterPerFrame,     // markers per frame
    //     board,                     // known board geometry
    //     imageSize,                 // image size
    //     cameraMatrix,              // output
    //     distCoeffs,                // output
    //     rvecs, tvecs, 0
    // );

    // Prints the outputs and saves the file
    std::cout << "Reprojection error: " << repError << "\n";
    std::cout << "Camera matrix:\n" << cameraMatrix << "\n";
    std::cout << "Dist coeffs:\n" << distCoeffs << "\n";

    // Save to YAML
    cv::FileStorage fs(fileName, cv::FileStorage::WRITE);
    fs << "camera_matrix" << cameraMatrix;
    fs << "dist_coeffs" << distCoeffs;
    fs << "reprojection_error" << repError;
    fs.release();                          

    return 0;
}