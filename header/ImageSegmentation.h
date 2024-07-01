#pragma once
#ifndef IMAGESEGMENTATION_H
#define IMAGESEGMENTATION_H

#include "libs.h"
#include "MorphologicalProcessing.h"

class ImageSegmentation
{
private:
    static void convertToGray(const cv::Mat &source_img, cv::Mat &dest_img);

    static bool isSimilar(const cv::Vec3f &color1, const cv::Vec3f &color2, float threshold);
    static bool isSimilar(float intensity1, float intensity2, float threshold);

    static bool isHomogeneous(const cv::Mat &region, int threshold);
    static void splitRegion(const cv::Mat &src, cv::Mat &labels, int x, int y, int width, int height, int &label, int threshold);

    static double calculateMeanIntensity(const cv::Mat &src, const cv::Mat &labels, int label);
    static bool shouldMerge(const cv::Mat &src, const cv::Mat &labels, int label1, int label2, int threshold);
    static void mergeRegions(cv::Mat &labels, int threshold);

    static void distanceTransform(const cv::Mat &binary, cv::Mat &distTransform);
    static void labelMarkers(const cv::Mat &sureFgr, cv::Mat &markers, cv::Mat &distTransform, int &label);
    static void applyWatershed(cv::Mat &markers, cv::Mat &distTransform);

public:
    /* Thresholding methods */
    static int globalThreshold(const cv::Mat &source_img, cv::Mat &dest_img, double thresh, double maxVal);
    static int adaptiveThreshold(const cv::Mat &source_img, cv::Mat &dest_img, double maxVal, int blockSize, double c);
    static int otsuThreshold(const cv::Mat &source_img, cv::Mat &dest_img);

    /* Regions-based methods */
    static bool isValidPoint(int x, int y, int rows, int cols);
    static int growRegions(const cv::Mat &source_img, cv::Mat &dest_img, const std::vector<cv::Point> &seeds, int threshold);

    static int regionSplitMerge(const cv::Mat &source_img, cv::Mat &dest_img, int threshold);

    /* Watersheds and K-means */
    static int watershed(const cv::Mat &source_img, cv::Mat &dest_img, bool flipBinary = false);
};

#endif