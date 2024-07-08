#include "ImageSegmentation.h"

std::vector<cv::Point> seedPts;
void onMouse(int event, int x, int y, int, void *param);

int main(int argc, char **argv)
{
    cv::Mat source_img = imread(argv[1], cv::IMREAD_COLOR);
    if (!source_img.data)
    {
        std::cout << "\n Image not found (wrong path) !";
        std::cout << "\n Path: " << argv[1];
        return 0;
    }

    cv::Mat dest_img;
    bool success = 0;

    if (str_compare(argv[3], "-threshold"))
    {
        if (str_compare(argv[4], "-global"))
        {
            success = ImageSegmentation::globalThreshold(source_img, dest_img, char_2_double(argv, 5), char_2_double(argv, 6));
        }
        else if (str_compare(argv[4], "-adaptive"))
        {
            success = ImageSegmentation::adaptiveThreshold(source_img, dest_img, char_2_double(argv, 5), char_2_int(argv, 6), char_2_double(argv, 7));
        }
        else if (str_compare(argv[4], "-otsu"))
        {
            success = ImageSegmentation::otsuThreshold(source_img, dest_img);
        }
    }
    else if (str_compare(argv[3], "-region"))
    {
        if (str_compare(argv[4], "-grow"))
        {
            cv::namedWindow("Select Seed Points");
            cv::setMouseCallback("Select Seed Points", onMouse, &source_img);
            cv::imshow("Select Seed Points", source_img);

            std::cout << "Click on the image to select seed points, then press any key to start region growing." << std::endl;
            cv::waitKey(0);

            if (seedPts.empty())
            {
                std::cerr << "No seed points selected." << std::endl;
                return -1;
            }

            success = ImageSegmentation::growRegions(source_img, dest_img, seedPts, char_2_int(argv, 5));
        }
        else if (str_compare(argv[4], "-spmr")) {
            success = ImageSegmentation::regionSplitMerge(source_img, dest_img, char_2_int(argv, 5));
        }
    }
    else if (str_compare(argv[3], "-watershed")) {
        bool flipBinary = char_2_int(argv, 4) == 1;
        success = ImageSegmentation::watershed(source_img, dest_img, flipBinary);
    }
    else if (str_compare(argv[3], "-kmeans")) {
        success = ImageSegmentation::kmeans(source_img, dest_img, char_2_int(argv, 4), char_2_int(argv, 5));
    }
    else if (str_compare(argv[3], "-meanshift")) {
        success = ImageSegmentation::meanShift(source_img, dest_img, char_2_double(argv, 4), char_2_double(argv, 5), char_2_int(argv, 6));
    }

    if (success)
    {
        imshow("Source image", source_img);
        imshow("Processed image", dest_img);
        imwrite(argv[2], dest_img);
    }
    else
        std::cout << "\n Something went wrong!";

    cv::waitKey(0);
    return 0;
}

void onMouse(int event, int x, int y, int, void *param)
{
    if (event == cv::EVENT_LBUTTONDOWN)
    {
        seedPts.emplace_back(x, y);
        cv::circle(*(cv::Mat *)param, cv::Point(x, y), 3, cv::Scalar(0, 0, 255), -1);
        cv::imshow("Select Seed Points", *(cv::Mat *)param);
    }
}