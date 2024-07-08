#include "KDTree.h"

KDTree::KDTree(const std::vector<cv::Point2f> &pts)
{
    std::vector<std::pair<cv::Point2f, int>> indexedPts(pts.size());

    for (size_t i = 0; i < pts.size(); i++)
    {
        indexedPts[i] = std::make_pair(pts[i], i);
    }

    root = buildTree(indexedPts, 0);
}

KDTree::~KDTree()
{
    destroyTree(root);
}

KDNode *KDTree::buildTree(std::vector<std::pair<cv::Point2f, int>> &points, int depth)
{
    if (points.empty())
        return nullptr;

    size_t axis = depth % 2;
    size_t median = points.size() / 2;
    std::nth_element(points.begin(), points.begin() + median, points.end(), [axis](const std::pair<cv::Point2f, int> &a, const std::pair<cv::Point2f, int> &b)
                     { return (axis == 0 ? a.first.x : a.first.y) < (axis == 0 ? b.first.x : b.first.y); });

    KDNode *node = new KDNode(points[median].first, points[median].second);
    std::vector<std::pair<cv::Point2f, int>> leftPts(points.begin(), points.begin() + median);
    std::vector<std::pair<cv::Point2f, int>> rightPts(points.begin() + median + 1, points.end());

    node->left = buildTree(leftPts, depth + 1);
    node->right = buildTree(rightPts, depth + 1);

    return node;
}

void KDTree::destroyTree(KDNode *node)
{
    if (node)
    {
        destroyTree(node->left);
        destroyTree(node->right);
        delete node;
    }
}

void KDTree::radiusSearch(const cv::Point2f &target, float radius, std::vector<int> &indices, std::vector<float> &distances)
{
    radiusSearchRecursive(root, target, radius * radius, indices, distances, 0);
}

void KDTree::radiusSearchRecursive(KDNode *node, const cv::Point2f &target, float radiusSquared, std::vector<int> &indices, std::vector<float> &distances, int depth)
{
    if (!node)
        return;

    float dist = squaredDistance(node->point, target);
    if (dist <= radiusSquared)
    {
        indices.push_back(node->index);
        distances.push_back(dist);
    }

    size_t axis = depth % 2;
    float diff = (axis == 0 ? target.x - node->point.x : target.y - node->point.y);
    float diffSquared = diff * diff;

    KDNode *nearChild = diff < 0 ? node->left : node->right;
    KDNode *farChild = diff < 0 ? node->right : node->left;

    radiusSearchRecursive(nearChild, target, radiusSquared, indices, distances, depth + 1);

    if (diffSquared < radiusSquared)
        radiusSearchRecursive(farChild, target, radiusSquared, indices, distances, depth);
}

float KDTree::squaredDistance(const cv::Point2f& p1, const cv::Point2f& p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return dx * dx + dy * dy;
}