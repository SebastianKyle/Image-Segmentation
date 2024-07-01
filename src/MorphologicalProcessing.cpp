#include "MorphologicalProcessing.h"

void MorphologicalProcessing::erode(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations)
{
    dest_img = source_img.clone();

    int width = source_img.cols, height = source_img.rows;

    for (int i = 0; i < iterations; i++)
    {
        cv::Mat temp = dest_img.clone();

        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                bool erodePixel = false;

                int kernelWidth = kernel.cols, kernelHeight = kernel.rows;

                for (int ky = 0; ky < kernelWidth; ky++)
                {
                    for (int kx = 0; kx < kernelHeight; kx++)
                    {
                        int ny = y + ky - kernelHeight / 2;
                        int nx = x + kx - kernelWidth / 2;

                        if (ny >= 0 && ny < height && nx >= 0 && nx < width)
                        {
                            if (kernel.at<uchar>(ky, kx) == 1 && source_img.at<uchar>(ny, nx) == 0)
                            {
                                erodePixel = true;
                                break;
                            }
                        }
                    }

                    if (erodePixel)
                        break;
                }

                temp.at<uchar>(y, x) = erodePixel ? 0 : 255;
            }
        }

        dest_img = temp;
    }
}

void MorphologicalProcessing::dilate(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations)
{
    dest_img = source_img.clone();

    int width = source_img.cols, height = source_img.rows;
    cv::Mat reflKernel;
    cv::flip(kernel, reflKernel, -1);

    for (int i = 0; i < iterations; i++) {
        cv::Mat temp = dest_img.clone();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                bool dilatePixel = false;

                int kernelWidth = kernel.cols, kernelHeight = kernel.rows;
                
                for (int ky = 0; ky < kernelHeight; ky++) {
                    for (int kx = 0; kx < kernelWidth; kx++) {
                        int ny = y - (ky - kernelHeight / 2);
                        int nx = x - (kx - kernelWidth / 2);

                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (reflKernel.at<uchar>(ky, kx) == 1 && source_img.at<uchar>(ny, nx) == 255) {
                                dilatePixel = true;
                                break;
                            }
                        }
                    }

                    if (dilatePixel) break;
                }

                temp.at<uchar>(y, x) = dilatePixel ? 255 : 0;
            }
        }

        dest_img = temp;
    }
}

void MorphologicalProcessing::open(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations) {
    cv::Mat temp;
    erode(source_img, temp, kernel, iterations);
    dilate(temp, dest_img, kernel, iterations);
}

void MorphologicalProcessing::close(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations) {
    cv::Mat temp;
    dilate(source_img, temp, kernel, iterations);
    erode(temp, dest_img, kernel, iterations);
}