#pragma once

#ifndef KDTREE_H
#define KDTREE_H

#include "libs.h"

struct KDNode
{
    cv::Point2f point;
    int index;
    KDNode *left;
    KDNode *right;

    KDNode(cv::Point2f pt, int idx) : point(pt), index(idx), left(nullptr), right(nullptr) {}
};

class KDTree
{
private:
    KDNode *root;

    KDNode *buildTree(std::vector<std::pair<cv::Point2f, int>> &points, int depth);
    void destroyTree(KDNode *node);
    void radiusSearchRecursive(KDNode *node, const cv::Point2f &target, float radiusSquared, std::vector<int> &indices, std::vector<float> &distances, int depth);

    static float squaredDistance(const cv::Point2f &p1, const cv::Point2f &p2);

public:
    KDTree(const std::vector<cv::Point2f> &pts);
    ~KDTree();

    void radiusSearch(const cv::Point2f &target, float radius, std::vector<int> &indices, std::vector<float> &distances);
};

#endif