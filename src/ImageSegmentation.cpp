#include "ImageSegmentation.h"

/*****************************************************************************************************************
 * Image Segmentation based on thresholding techniques
 */

void ImageSegmentation::convertToGray(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (source_img.channels() == 3)
    {
        cv::cvtColor(source_img, dest_img, cv::COLOR_BGR2GRAY);
    }
    else
    {
        dest_img = source_img.clone();
    }
}

int ImageSegmentation::globalThreshold(const cv::Mat &source_img, cv::Mat &dest_img, double thresh, double maxVal)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat grayImg, threshImg;
    convertToGray(source_img, grayImg);
    threshImg = cv::Mat::zeros(grayImg.size(), CV_8UC1);

    int width = grayImg.cols, height = grayImg.rows;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (grayImg.at<uchar>(y, x) > thresh)
            {
                threshImg.at<uchar>(y, x) = cv::saturate_cast<uchar>(maxVal);
            }
            else
            {
                threshImg.at<uchar>(y, x) = 0;
            }
        }
    }

    dest_img = threshImg.clone();

    return 1;
}

int ImageSegmentation::adaptiveThreshold(const cv::Mat &source_img, cv::Mat &dest_img, double maxVal, int blockSize, double c)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat gray, threshImg;
    convertToGray(source_img, gray);
    threshImg = cv::Mat::zeros(gray.size(), CV_8UC1);

    int width = gray.cols, height = gray.rows;
    int halfBlockSize = blockSize / 2;
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int xStart = std::max(0, x - halfBlockSize);
            int xEnd = std::min(width - 1, x + halfBlockSize);
            int yStart = std::max(0, y - halfBlockSize);
            int yEnd = std::min(height - 1, y + halfBlockSize);

            double sum = 0;
            int count = 0;
            for (int j = yStart; j <= yEnd; j++)
            {
                for (int i = xStart; i <= xEnd; i++)
                {
                    sum += gray.at<uchar>(j, i);
                    count++;
                }
            }

            double mean = sum / count;
            if (gray.at<uchar>(y, x) > mean - c)
            {
                threshImg.at<uchar>(y, x) = cv::saturate_cast<uchar>(maxVal);
            }
            else
            {
                threshImg.at<uchar>(y, x) = 0;
            }
        }
    }

    dest_img = threshImg.clone();

    return 1;
}

int ImageSegmentation::otsuThreshold(const cv::Mat &source_img, cv::Mat &dest_img)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat gray, threshImg;
    convertToGray(source_img, gray);
    threshImg = cv::Mat::zeros(gray.size(), CV_8UC1);

    int width = gray.cols, height = gray.rows;

    // Compute histogram
    const int histSize = 256;
    int hist[histSize] = {0};
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            hist[gray.at<uchar>(y, x)]++;
        }
    }

    int totalPixels = width * height;

    float sum = 0;
    for (int i = 0; i < histSize; i++)
    {
        sum += i * hist[i];
    }

    float sumB = 0;
    int wB = 0;
    int wF = 0;

    float varMax = 0;
    int threshold = 0;

    // Find optimal t that produces maximum between class variance
    for (int t = 0; t < histSize; t++)
    {
        wB += hist[t];
        if (wB == 0)
            continue;

        wF = totalPixels - wB;
        if (wF == 0)
            break;

        sumB += static_cast<float>(t * hist[t]);

        float mB = sumB / wB;
        float mF = (sum - sumB) / wF;

        float varBetween = static_cast<float>(wB) * static_cast<float>(wF) * (mB - mF) * (mB - mF);
        if (varBetween > varMax)
        {
            varMax = varBetween;
            threshold = t;
        }
    }

    // Apply threshold
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            if (gray.at<uchar>(y, x) > threshold)
            {
                threshImg.at<uchar>(y, x) = 255;
            }
            else
            {
                threshImg.at<uchar>(y, x) = 0;
            }
        }
    }

    dest_img = threshImg.clone();
    return 1;
}

/*****************************************************************************************************************
 * Image Segmentation by region-based techniques
 */

bool ImageSegmentation::isValidPoint(int x, int y, int rows, int cols)
{
    return (x >= 0 && x < cols && y >= 0 && y < rows);
}

bool ImageSegmentation::isSimilar(const cv::Vec3f &color1, const cv::Vec3f &color2, float threshold)
{
    float diff = cv::norm(color1 - color2);
    return diff < threshold;
}

bool ImageSegmentation::isSimilar(float intensity1, float intensity2, float threshold)
{
    return std::abs(intensity1 - intensity2) < threshold;
}

int ImageSegmentation::growRegions(const cv::Mat &source_img, cv::Mat &dest_img, const std::vector<cv::Point> &seeds, int threshold)
{
    if (!source_img.data)
    {
        return 0;
    }

    bool isColor = (source_img.channels() == 3);

    cv::Mat gray, labImg, segmented = cv::Mat::zeros(source_img.size(), CV_32S);

    if (isColor)
    {
        cv::cvtColor(source_img, labImg, cv::COLOR_BGR2Lab);
        labImg.convertTo(labImg, CV_32F);
    }
    else
    {
        cv::cvtColor(source_img, gray, cv::COLOR_BGR2GRAY);
    }

    int width = labImg.cols, height = labImg.rows;

    std::queue<cv::Point> pixelQueue;
    std::vector<cv::Vec3f> seedColors;
    std::vector<float> seedInts;

    // Initialize queue with seed points
    for (size_t i = 0; i < seeds.size(); i++)
    {
        pixelQueue.push(seeds[i]);

        if (isColor)
        {
            seedColors.push_back(labImg.at<cv::Vec3f>(seeds[i]));
        }
        else
        {
            seedInts.push_back(gray.at<uchar>(seeds[i]));
        }

        // Label regions
        segmented.at<int>(seeds[i]) = static_cast<int>(i) + 1;
    }

    int pixelAssigned = 0;
    while (!pixelQueue.empty())
    {
        cv::Point p = pixelQueue.front();
        pixelQueue.pop();

        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                int newX = p.x + dx;
                int newY = p.y + dy;

                if (isValidPoint(newX, newY, height, width) && segmented.at<int>(newY, newX) == 0)
                {
                    bool withinThreshold = false;
                    int label = segmented.at<int>(p.y, p.x);

                    if (isColor)
                    {
                        cv::Vec3f newColor = labImg.at<cv::Vec3f>(newY, newX);
                        if (isSimilar(newColor, seedColors[label - 1], threshold))
                        {
                            withinThreshold = true;
                        }
                    }
                    else
                    {
                        float newInt = gray.at<uchar>(newY, newX);
                        if (isSimilar(newInt, seedInts[label - 1], threshold))
                        {
                            withinThreshold = true;
                        }
                    }

                    if (withinThreshold)
                    {
                        segmented.at<int>(newY, newX) = label;
                        pixelQueue.push(cv::Point(newX, newY));

                        pixelAssigned++;
                    }
                }
            }
        }
    }

    std::cout << "\n Pixels assigned: " << pixelAssigned << std::endl;

    cv::Mat segmentedColor = cv::Mat::zeros(source_img.size(), CV_8UC3);
    for (int y = 0; y < segmented.rows; y++)
    {
        for (int x = 0; x < segmented.cols; x++)
        {
            int label = segmented.at<int>(y, x);
            if (label > 0)
            {
                segmentedColor.at<cv::Vec3b>(y, x) = cv::Vec3b((label * 50) % 255, (label * 100) % 255, (label * 150) % 255);
            }
        }
    }

    dest_img = segmentedColor.clone();
    return 1;
}

bool ImageSegmentation::isHomogeneous(const cv::Mat &region, int threshold)
{
    cv::Scalar mean, stdDev;
    cv::meanStdDev(region, mean, stdDev);

    double range = std::abs(mean[0] - stdDev[0]);

    return range < threshold;
}

void ImageSegmentation::splitRegion(const cv::Mat &src, cv::Mat &labels, int x, int y, int width, int height, int &label, int threshold)
{
    cv::Rect region(x, y, width, height);
    cv::Mat subRegion = src(region);

    if (isHomogeneous(subRegion, threshold))
    {
        labels(region).setTo(label);
        label++;
    }
    else
    {
        int halfWidth = width / 2;
        int halfHeight = height / 2;

        if (halfWidth > 0 && halfHeight > 0)
        {
            splitRegion(src, labels, x, y, halfWidth, halfHeight, label, threshold);
            splitRegion(src, labels, x + halfWidth, y, halfWidth, halfHeight, label, threshold);
            splitRegion(src, labels, x + halfWidth, y + halfHeight, halfWidth, halfHeight, label, threshold);
            splitRegion(src, labels, x, y + halfHeight, halfWidth, halfHeight, label, threshold);
        }
        else
        {
            labels(region).setTo(label);
            label++;
        }
    }
}

double ImageSegmentation::calculateMeanIntensity(const cv::Mat &src, const cv::Mat &labels, int label)
{
    cv::Scalar meanInt = cv::mean(src, labels == label);
    return meanInt[0];
}

bool ImageSegmentation::shouldMerge(const cv::Mat &src, const cv::Mat &labels, int label1, int label2, int threshold)
{
    double mean1 = calculateMeanIntensity(src, labels, label1);
    double mean2 = calculateMeanIntensity(src, labels, label2);

    return std::abs(mean1 - mean2) < threshold;
}

void ImageSegmentation::mergeRegions(cv::Mat &labels, int threshold)
{
    int height = labels.rows, width = labels.cols;
    int currLabel = 1;
    std::unordered_map<int, int> labelMap;

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int label = labels.at<int>(y, x);
            if (label == 0)
                continue;

            std::vector<cv::Point> neighbors = {
                {x - 1, y}, {x + 1, y}, {x, y - 1}, {x, y + 1}, {x - 1, y - 1}, {x + 1, y - 1}, {x - 1, y + 1}, {x + 1, y + 1}};

            for (const auto &neighbor : neighbors)
            {
                int nx = neighbor.x;
                int ny = neighbor.y;

                if (nx >= 0 && nx < width && ny >= 0 && ny < height)
                {
                    int neighborLabel = labels.at<int>(ny, nx);

                    if (neighborLabel != 0 && neighborLabel != label)
                    {
                        if (shouldMerge(labels, labels, label, neighborLabel, threshold))
                        {
                            int minLabel = std::min(label, neighborLabel);
                            int maxLabel = std::max(label, neighborLabel);

                            if (labelMap.find(maxLabel) == labelMap.end())
                            {
                                labelMap[maxLabel] = minLabel;
                            }
                            else
                            {
                                labelMap[maxLabel] = std::min(labelMap[maxLabel], minLabel);
                            }
                        }
                    }
                }
            }
        }
    }

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int label = labels.at<int>(y, x);

            while (labelMap.find(label) != labelMap.end())
            {
                label = labelMap[label];
            }

            labels.at<int>(y, x) = label;
        }
    }
}

int ImageSegmentation::regionSplitMerge(const cv::Mat &source_img, cv::Mat &dest_img, int threshold)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat gray;
    convertToGray(source_img, gray);

    cv::Mat labels = cv::Mat::zeros(gray.size(), CV_32S);
    int label = 1;

    splitRegion(gray, labels, 0, 0, gray.cols, gray.rows, label, threshold);
    mergeRegions(labels, threshold);

    cv::Mat segmentedColor = cv::Mat::zeros(gray.size(), CV_8UC3);
    std::vector<cv::Vec3b> colors(label);

    for (int i = 0; i < label; i++)
    {
        colors[i] = cv::Vec3b((i * 50) % 255, (i * 100) % 255, (i * 150) % 255);
    }

    for (int y = 0; y < labels.rows; y++)
    {
        for (int x = 0; x < labels.cols; x++)
        {
            int l = labels.at<int>(y, x);

            if (l > 0)
            {
                segmentedColor.at<cv::Vec3b>(y, x) = colors[l - 1];
            }
        }
    }

    dest_img = segmentedColor.clone();

    return 1;
}

/*****************************************************************************************************************
 * Image Segmentation using watershed algorithm
 */

/*****************************************************************************************************************
 * Distance Transform
 * 
 * Obtain distance of foreground pixels to the nearest background pixel
 */
void ImageSegmentation::distanceTransform(const cv::Mat &binary, cv::Mat &distTransform)
{
    const float INF = std::numeric_limits<float>::max();
    distTransform = cv::Mat::zeros(binary.size(), CV_32FC1);

    // Initialize distance transform for foreground and background
    for (int y = 0; y < binary.rows; y++)
    {
        for (int x = 0; x < binary.cols; x++)
        {
            if (binary.at<uchar>(y, x) == 255)
            {
                distTransform.at<float>(y, x) = INF; // Foreground pixel
            }
            else
            {
                distTransform.at<float>(y, x) = 0.0f; // Background pixel
            }
        }
    }

    // First pass: scan from top-left to bottom-right
    for (int y = 0; y < binary.rows; y++)
    {
        for (int x = 0; x < binary.cols; x++)
        {
            if (binary.at<uchar>(y, x) == 255)
            {
                // Check the neighbors in the previous row and the previous column
                if (y > 0)
                    distTransform.at<float>(y, x) = std::min(distTransform.at<float>(y, x), distTransform.at<float>(y - 1, x) + 1);
                if (x > 0)
                    distTransform.at<float>(y, x) = std::min(distTransform.at<float>(y, x), distTransform.at<float>(y, x - 1) + 1);
            }
        }
    }

    // Second pass: scan from bottom-right to top-left
    for (int y = binary.rows - 1; y >= 0; y--)
    {
        for (int x = binary.cols - 1; x >= 0; x--)
        {
            if (binary.at<uchar>(y, x) == 255)
            {
                // Check the neighbors in the next row and the next column
                if (y < binary.rows - 1)
                    distTransform.at<float>(y, x) = std::min(distTransform.at<float>(y, x), distTransform.at<float>(y + 1, x) + 1);
                if (x < binary.cols - 1)
                    distTransform.at<float>(y, x) = std::min(distTransform.at<float>(y, x), distTransform.at<float>(y, x + 1) + 1);
            }
        }
    }

    cv::normalize(distTransform, distTransform, 0, 1.0, cv::NORM_MINMAX);

    imshow("Distance: ", distTransform);
}

/*****************************************************************************************************************
 * Label internal markers
 * 
 * Label the internal markers for objects using the foreground pixels
 */
void ImageSegmentation::labelMarkers(const cv::Mat &sureFgr, cv::Mat &markers, cv::Mat &distTransform, int &label) {
    for (int y = 0; y < sureFgr.rows; y++)
    {
        for (int x = 0; x < sureFgr.cols; x++)
        {
            if (sureFgr.at<uchar>(y, x) == 255 && markers.at<int>(y, x) == 0)
            {
                std::queue<cv::Point> q;
                q.push(cv::Point(x, y));
                markers.at<int>(y, x) = label;

                while (!q.empty())
                {
                    cv::Point p = q.front();
                    q.pop();

                    for (int i = -1; i <= 1; i++)
                    {
                        for (int j = -1; j <= 1; j++)
                        {
                            int nx = p.x + j;
                            int ny = p.y + i;

                            if (nx >= 0 && nx < sureFgr.cols && ny >= 0 && ny < sureFgr.rows &&
                                sureFgr.at<uchar>(ny, nx) == 255 && markers.at<int>(ny, nx) == 0 && 
                                distTransform.at<float>(ny, nx) >= 0.1)
                            {
                                markers.at<int>(ny, nx) = label;
                                q.push(cv::Point(nx, ny));
                            }
                        }
                    }
                }

                label++;
            }
        }
    }
}

/*****************************************************************************************************************
 * Apply Watershed 
 * 
 * Populate internal markers and define watershed lines
 */
void ImageSegmentation::applyWatershed(cv::Mat &markers, cv::Mat &distTransform)
{
    using Pixel = std::pair<int, int>;
    auto cmp = [&distTransform](const Pixel &a, const Pixel &b)
    {
        return distTransform.at<float>(a.first, a.second) > distTransform.at<float>(b.first, b.second);
    };
    std::priority_queue<Pixel, std::vector<Pixel>, decltype(cmp)> pq(cmp);

    for (int y = 0; y < markers.rows; ++y)
    {
        for (int x = 0; x < markers.cols; ++x)
        {
            if (markers.at<int>(y, x) > 1)
            {
                pq.push({y, x});
            }
        }
    }

    std::vector<cv::Point> directions = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}, {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

    while (!pq.empty())
    {
        Pixel p = pq.top();
        pq.pop();
        int y = p.first, x = p.second;

        for (const auto &d : directions)
        {
            int ny = y + d.y, nx = x + d.x;

            if (ny >= 0 && ny < markers.rows && nx >= 0 && nx < markers.cols)
            {
                if (markers.at<int>(ny, nx) == 1)
                {
                    if (distTransform.at<float>(ny, nx) >= 0.15) {
                        markers.at<int>(ny, nx) = markers.at<int>(y, x);
                        pq.push({ny, nx});
                    }
                }
                else if (markers.at<int>(ny, nx) == 0)
                {
                    markers.at<int>(ny, nx) = -1; // Watershed line
                }
            }
        }
    }
}

/*****************************************************************************************************************
 * Watershed Segmentation
 * 
 * Combine functionalities to produce segmented image:
 * 
 *  -> Apply Otsu threshold (may flip based on user decision) 
 *  -> Apply morphological filters to obtain "sure background" and "sure foreground" images
 *  -> Compute distance transform image for thresholded image
 *  -> Label the internal markers based on foreground image and distance transform image
 *  -> Apply watershed to determine outer markers
 *  -> Generate colors and produce segmented image 
 */
int ImageSegmentation::watershed(const cv::Mat &source_img, cv::Mat &dest_img, bool flipBinary)
{
    if (!source_img.data)
    {
        return 0;
    }

    cv::Mat gray, binary;
    convertToGray(source_img, gray);

    otsuThreshold(source_img, binary);
    if (flipBinary) {
        cv::bitwise_not(binary, binary);
    }

    // Remove noise using morphological open operator
    cv::Mat kernel = cv::Mat::ones(3, 3, CV_8U);
    MorphologicalProcessing::open(binary, binary, kernel, 3);
    MorphologicalProcessing::dilate(binary, binary, kernel, 1);

    // Find sure background area
    cv::Mat sureBgr;
    MorphologicalProcessing::dilate(binary, sureBgr, kernel, 3);

    // Distance transform
    cv::Mat distTransform;
    distanceTransform(binary, distTransform);

    cv::Mat distThresh;
    cv::threshold(distTransform, distThresh, 0.2, 1.0, cv::THRESH_BINARY);
    cv::Mat dist8U;
    distThresh.convertTo(dist8U, CV_8U, 255.0);
    MorphologicalProcessing::dilate(dist8U, dist8U, kernel, 2);

    cv::Mat sureFgr;
    dist8U.convertTo(sureFgr, CV_8U, 255.0);
    MorphologicalProcessing::open(sureFgr, sureFgr, kernel, 5);

    cv::Mat unknown = sureBgr - sureFgr;
    MorphologicalProcessing::open(unknown, unknown, kernel, 3);

    imshow("binary", binary);
    imshow("sureBgr", sureBgr);
    imshow("sureFgr", sureFgr);
    imshow("unknown", unknown);

    // Label internal markers
    cv::Mat markers = cv::Mat::zeros(sureFgr.size(), CV_32S);
    int label = 2;
    labelMarkers(sureFgr, markers, distTransform, label);

    // markers = markers + 2;
    markers.setTo(1, unknown);

    // Watershed
    applyWatershed(markers, distTransform);

    std::vector<cv::Vec3b> colors(label + 1);
    colors[0] = cv::Vec3b(0, 0, 0); // background
    colors[1] = cv::Vec3b(0, 0, 0); // unknown
    for (int i = 2; i <= label; i++)
    {
        colors[i] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
    }

    dest_img = cv::Mat::zeros(source_img.size(), CV_8UC3);
    for (int y = 0; y < markers.rows; ++y)
    {
        for (int x = 0; x < markers.cols; ++x)
        {
            int markerLabel = markers.at<int>(y, x);

            if (markerLabel >= 0 && markerLabel <= label)
            {
                dest_img.at<cv::Vec3b>(y, x) = colors[markerLabel];
            }
            else
            {
                dest_img.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255); // watershed lines
            }
        }
    }

    return 1;
}