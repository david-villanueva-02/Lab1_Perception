#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/objdetect/aruco_board.hpp>

#include <iostream>
#include <string>
#include <map>
#include <sstream>

// Dictionary that maps the string input to the actual dictionary
static int dictFromString(const std::string& name) {
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

int main(int argc, char* argv[]) {
    if (argc < 7) {
        std::cout << "Usage: ./generate_board <rows> <cols> <dictionary> <marker_size_px> <separation_px> <file_name>";
        return -1;
    }

    // --- Extract parameters ----
    const int boardRows = std::atoi(argv[1]);
    const int boardCols = std::atoi(argv[2]);
    const int dictionary_name = dictFromString(argv[3]);
    const int markerSize = std::atoi(argv[4]);
    const int separation = std::atoi(argv[5]);
    const std::string fileName = argv[6];

    // Dictionary
    const auto dictionary = cv::aruco::getPredefinedDictionary(dictionary_name);

    auto board = cv::makePtr<cv::aruco::GridBoard>(
        cv::Size(boardCols, boardRows),  // (cols, rows)
        markerSize,
        separation,
        dictionary
    );

    // Output image size 
    const int width = boardCols * markerSize + (boardCols + 1) * separation;
    const int height = boardRows * markerSize + (boardRows + 1) * separation;

    // Create the base image and the board over it
    cv::Mat boardImage;
    board->generateImage(cv::Size(width, height), boardImage, separation, 1);

    // Save the file
    std::stringstream ss;
    ss << "/home/david/Documents/IFRoS/Perception/labs/lab1/images/" << fileName;
    if (!cv::imwrite(ss.str(), boardImage)) {
        std::cerr << "Failed to write: " << ss.str() << "\n";
        return -1;
    }

    return 0;
}