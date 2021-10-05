/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include <algorithm>
#include <iostream>
#include <vector>

namespace Saiga
{
// D : Dimension. for example D=3 for 3 dimensional points
// point_t : should be a vector type. for example vec2 or vec3
template <int D, typename point_t>
class SAIGA_TEMPLATE KDTree
{
   public:
    // create an empty tree
    KDTree() {}
    KDTree(const std::vector<point_t>& points);

    // returns the nearest point in this tree to the searchpoint
    int NearestNeighborSearch(const point_t& searchPoint);

    // returns the k nearest points in this tree to the searchpoint
    std::vector<int> KNearestNeighborSearch(const point_t& searchPoint, int k);


    std::vector<int> RadiusSearch(const point_t& searchPoint, float radius);



   private:
    typedef int index_t;
    typedef unsigned int axis_t;
    typedef std::vector<std::pair<float, index_t>> queue_t;

    struct kd_node_t
    {
        point_t p;
        int initial_index;
        index_t left = -1, right = -1;
    };

    std::vector<kd_node_t> nodes;
    index_t rootNode = -1;

    index_t sortByAxis(index_t startIndex, index_t endIndex, axis_t axis);
    index_t make_tree(index_t startIndex, index_t endIndex, axis_t currentAxis);

    // rekursive helper functions for nearest neighbour lookup
    void NearestNeighborSearch(index_t currentNode, const point_t& searchPoint, axis_t currentAxis, index_t& bestNode,
                               float& bestDist);

    void KNearestNeighborSearch(index_t currentNode, const point_t& searchPoint, int k, axis_t currentAxis,
                                queue_t& queue);

    void RadiusSearch(index_t currentNode, const point_t& searchPoint, float r, axis_t currentAxis,
                      std::vector<int>& result);

    float addToQueue(queue_t& queue, index_t currentNode, float distance);
    float distance(point_t a, point_t b);
    void printPoints(index_t startIndex, index_t endIndex);
};

template <int D, typename point_t>
KDTree<D, point_t>::KDTree(const std::vector<point_t>& points)
{
    nodes.resize(points.size());
    for (int i = 0; i < points.size(); ++i)
    {
        nodes[i].p             = points[i];
        nodes[i].initial_index = i;
    }
    rootNode = make_tree(0, nodes.size(), 0);
}

template <int D, typename point_t>
typename KDTree<D, point_t>::index_t KDTree<D, point_t>::sortByAxis(index_t startIndex, index_t endIndex, axis_t axis)
{
    auto cmp = [axis](const kd_node_t& a, const kd_node_t& b) -> bool { return a.p[axis] < b.p[axis]; };
    std::sort(nodes.begin() + startIndex, nodes.begin() + endIndex, cmp);
    // return the median point
    return (startIndex + endIndex) / 2;
}

template <int D, typename point_t>
typename KDTree<D, point_t>::index_t KDTree<D, point_t>::make_tree(index_t startIndex, index_t endIndex,
                                                                   axis_t currentAxis)
{
    if (startIndex == endIndex) return -1;
    if (startIndex + 1 == endIndex) return startIndex;

    index_t median = sortByAxis(startIndex, endIndex, currentAxis);
    currentAxis    = (currentAxis + 1) % D;


    nodes[median].left  = make_tree(startIndex, median, currentAxis);
    nodes[median].right = make_tree(median + 1, endIndex, currentAxis);

    return median;
}

template <int D, typename point_t>
int KDTree<D, point_t>::NearestNeighborSearch(const point_t& searchPoint)
{
    KDTree::index_t bestNode;
    float bestDist = std::numeric_limits<float>::infinity();
    NearestNeighborSearch(rootNode, searchPoint, 0, bestNode, bestDist);
    return nodes[bestNode].initial_index;
}


template <int D, typename point_t>
void KDTree<D, point_t>::NearestNeighborSearch(index_t currentNode, const point_t& searchPoint, axis_t currentAxis,
                                               index_t& bestNode, float& bestDist)
{
    if (currentNode == -1) return;

    // calculate distance to current point and update the current best
    float d = distance(nodes[currentNode].p, searchPoint);
    if (d < bestDist)
    {
        bestDist = d;
        bestNode = currentNode;
    }

    // exact match (can't get any better)
    if (d == 0) return;

    // the (signed) distance of the searchpoint to the current split axis
    float dAxis = nodes[currentNode].p[currentAxis] - searchPoint[currentAxis];
    // the actual distance to the point is squared so we also need to square the distance to the axis
    float dAxisSquared = dAxis * dAxis;

    currentAxis = (currentAxis + 1) % D;

    // first traverse the subtree in which the point lays
    NearestNeighborSearch(dAxis > 0 ? nodes[currentNode].left : nodes[currentNode].right, searchPoint, currentAxis,
                          bestNode, bestDist);

    // when the distance to the axis is greater than the current distance
    // we don't need to traverse the other sub tree
    if (dAxisSquared >= bestDist) return;

    // there may be a better point in this subtree
    NearestNeighborSearch(dAxis > 0 ? nodes[currentNode].right : nodes[currentNode].left, searchPoint, currentAxis,
                          bestNode, bestDist);
}


template <int D, typename point_t>
std::vector<int> KDTree<D, point_t>::KNearestNeighborSearch(const point_t& searchPoint, int k)
{
    queue_t queue(k);
    for (auto& p : queue)
    {
        p.second = -1;
        p.first  = 1e10;
    }
    KNearestNeighborSearch(rootNode, searchPoint, k, 0, queue);

    std::vector<int> points;
    for (auto& p : queue)
    {
        if (p.second != -1)
        {
            points.push_back(nodes[p.second].initial_index);
        }
    }
    return points;
}


template <int D, typename point_t>
void KDTree<D, point_t>::KNearestNeighborSearch(index_t currentNode, const point_t& searchPoint, int k,
                                                axis_t currentAxis, queue_t& queue)
{
    if (currentNode == -1) return;

    // calculate distance to current point and update the current best
    float d     = distance(nodes[currentNode].p, searchPoint);
    float lastD = addToQueue(queue, currentNode, d);


    // the (signed) distance of the searchpoint to the current split axis
    float dAxis = nodes[currentNode].p[currentAxis] - searchPoint[currentAxis];
    // the actual distance to the point is squared so we also need to square the distance to the axis
    float dAxisSquared = dAxis * dAxis;

    currentAxis = (currentAxis + 1) % D;

    // first traverse the subtree in which the point lays
    KNearestNeighborSearch(dAxis > 0 ? nodes[currentNode].left : nodes[currentNode].right, searchPoint, k, currentAxis,
                           queue);

    // when the distance to the axis is greater than the current distance
    // we don't need to traverse the other sub tree
    if (dAxisSquared >= lastD) return;

    // there may be a better point in this subtree
    KNearestNeighborSearch(dAxis > 0 ? nodes[currentNode].right : nodes[currentNode].left, searchPoint, k, currentAxis,
                           queue);

    //    nearestNeighbour(dAxis > 0 ? nodes[currentNode].right : nodes[currentNode].left, searchPoint, currentAxis
    //    ,bestNode,  bestDist);
}

template <int D, typename point_t>
float KDTree<D, point_t>::addToQueue(queue_t& queue, index_t currentNode, float distance)
{
    float lastD = queue[queue.size() - 1].first;
    if (distance >= lastD) return lastD;

    queue[queue.size() - 1].first  = distance;
    queue[queue.size() - 1].second = currentNode;

    std::sort(queue.begin(), queue.end());


    return queue[queue.size() - 1].first;
}


template <int D, typename point_t>
void KDTree<D, point_t>::RadiusSearch(index_t currentNode, const point_t& searchPoint, float r, axis_t currentAxis,
                                      std::vector<int>& result)
{
    if (currentNode == -1) return;

    // calculate distance to current point and update the current best
    float d = distance(nodes[currentNode].p, searchPoint);
    if (d < r)
    {
        result.push_back(nodes[currentNode].initial_index);
    }


    // the (signed) distance of the searchpoint to the current split axis
    float dAxis = nodes[currentNode].p[currentAxis] - searchPoint[currentAxis];
    // the actual distance to the point is squared so we also need to square the distance to the axis
    float dAxisSquared = dAxis * dAxis;

    currentAxis = (currentAxis + 1) % D;

    // first traverse the subtree in which the point lies
    RadiusSearch(dAxis > 0 ? nodes[currentNode].left : nodes[currentNode].right, searchPoint, r, currentAxis, result);

    // when the distance to the axis is greater than the current distance
    // we don't need to traverse the other sub tree
    if (dAxisSquared >= r) return;

    // there may be a better point in this subtree
    RadiusSearch(dAxis > 0 ? nodes[currentNode].right : nodes[currentNode].left, searchPoint, r, currentAxis, result);
}


template <int D, typename point_t>
std::vector<int> KDTree<D, point_t>::RadiusSearch(const point_t& searchPoint, float r)
{
    std::vector<int> points;
    RadiusSearch(rootNode, searchPoint, r * r, 0, points);
    std::sort(points.begin(), points.end());
    return points;
}



template <int D, typename point_t>
float KDTree<D, point_t>::distance(point_t a, point_t b)
{
    // use the squared distance so we don't have to calculate the sqrt
    point_t tmp = a - b;
    return dot(tmp, tmp);
}

}  // namespace Saiga
