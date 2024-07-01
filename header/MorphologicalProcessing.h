#pragma once
#ifndef MORPHOLOGICALPROCESSING_H
#define MORPHOLOGICALPROCESSING_H

#include "libs.h"

class MorphologicalProcessing
{
private:
public:
    static void erode(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations = 1);
    static void dilate(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations = 1);
    static void open(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations = 1);
    static void close(const cv::Mat &source_img, cv::Mat &dest_img, const cv::Mat &kernel, int iterations = 1);
};

#endif