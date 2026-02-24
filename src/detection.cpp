#include "opencv2/core/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/objdetect/aruco_detector.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <video source no.>" << std::endl;
        return -1;
    }

    // std::cout << std::atoi(argv[1]) << std::endl;
    cv::VideoCapture webCam(std::atoi(argv[1])); // VideoCapture object declaration. Usually 0 is the integrated, 2 is the first external USB one↪→

    if (webCam.isOpened() == false)
    { // Check if the VideoCapture object has been correctly associated to the webcam↪→
        std::cerr << "error: Webcam could not be connected." << std::endl;
        return -1;
    }

    cv::Mat imgOriginal;  // input image

    char charCheckForESCKey{0};

    cv::namedWindow("imgOriginal");

    while (charCheckForESCKey != 27 && webCam.isOpened())
    {                                                 // loop until ESC key is pressed or webcam is lost
        bool frameSuccess = webCam.read(imgOriginal); // get next frame from input stream

        if (!frameSuccess || imgOriginal.empty())
        { // if the frame was not read or read wrongly
            std::cerr << "error: Frame could not be read." << std::endl;
            break;
        }

        // create matrix to store the marker corners and ids
        std::vector<int> markerIds;
        std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
        
        // create the parametes for the detector and get the dictionary to use.
        cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
        cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);
        cv::aruco::ArucoDetector detector(dictionary, detectorParams);

        // detect the markers on the imageand draw the markers detected
        detector.detectMarkers(imgOriginal, markerCorners, markerIds, rejectedCandidates);
        cv::Mat outputImage = imgOriginal.clone();
        cv::aruco::drawDetectedMarkers(outputImage, markerCorners, markerIds);

        // Show output video results windows
        cv::imshow("imgOriginal", outputImage);

        charCheckForESCKey = cv::waitKey(1); // gets the key pressed
    }
    return 0;
}