/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"

#include <vector>
#include <fstream>

namespace Saiga
{
template <typename T>
class WeightedUndirectedGraph
{
   public:
    static constexpr T invalid_weight = std::numeric_limits<T>::max();

    WeightedUndirectedGraph(int n = 0) { Resize(n); }

    void Resize(int n)
    {
        this->n = n;
        adjacencyMatrix.resize(n, n);
        Clear();
    }

    void Clear() { adjacencyMatrix.setConstant(invalid_weight); }

    struct Edge
    {
        int from, to;
        double weight;
    };

    void AddEdge(const Edge& edge) { AddEdge(edge.from, edge.to, edge.weight); }

    void AddEdge(int from, int to, T weight)
    {
        adjacencyMatrix(from, to) = weight;
        adjacencyMatrix(to, from) = weight;
    }



    void BuildMST()
    {
        // Efficient Implementation of Prim's algorithm.
        // Starting at any vertex, the  edge with the highest weight is added to the graph.
        // This is done until all vertices are added or we detect that the graph is not connected.


        // mark nodes already added
        visited.resize(n);
        std::fill(visited.begin(), visited.end(), false);

        parents.resize(n);
        std::fill(parents.begin(), parents.end(), -1);

        current_weight.resize(n);
        std::fill(current_weight.begin(), current_weight.end(), std::numeric_limits<T>::min());

        // Set weight of node 0 so it will be selected in the first iteration
        // Don't set visited otherwise it would not be selected.
        parents[0]        = -1;
        current_weight[0] = 1;

        for (int i = 0; i < n - 1; ++i)
        {
            // find node with largest weight
            int next_node   = -1;
            int best_weight = -1;
            for (int j = 0; j < n; ++j)
            {
                // already added to the graph
                if (visited[j]) continue;

                if (current_weight[j] > best_weight)
                {
                    best_weight = current_weight[j];
                    next_node   = j;
                }
            }

            // The graph is not connected
            if (next_node == -1) break;

            SAIGA_ASSERT(next_node != -1);

            visited[next_node] = true;

            // update weight of adjacency nodes
            for (int j = 0; j < n; ++j)
            {
                if (visited[j]) continue;
                auto v = adjacencyMatrix(next_node, j);
                if (v != invalid_weight && v > current_weight[j])
                {
                    current_weight[j] = v;
                    parents[j]        = next_node;
                }
            }
        }
    }



    WeightedUndirectedGraph GetMSTAsGraph()
    {
        WeightedUndirectedGraph output(n);
        for (int i = 1; i < n; i++)
        {
            if (parents[i] >= 0)
            {
                Edge e;
                e.from   = parents[i];
                e.to     = i;
                e.weight = adjacencyMatrix(i, parents[i]);
                output.AddEdge(e);
            }
        }

        return output;
    }

    std::vector<Edge> GetMSTEdgesForNode(int node)
    {
        std::vector<WeightedUndirectedGraph::Edge> result;

        for (int i = 1; i < n; i++)
        {
            if (parents[i] >= 0)
            {
                if (i == node)
                {
                    Edge e;
                    e.from   = i;
                    e.to     = parents[i];
                    e.weight = adjacencyMatrix(i, parents[i]);
                    result.push_back(e);
                }
                else if (parents[i] == node)
                {
                    Edge e;
                    e.from   = parents[i];
                    e.to     = i;
                    e.weight = adjacencyMatrix(i, parents[i]);
                    result.push_back(e);
                }
            }
        }
        return result;
    }

    T WeightOfWeakestMSTEdge()
    {
        T w = std::numeric_limits<T>::max();
        for (int i = 1; i < n; i++)
        {
            if (parents[i] >= 0)
            {
                w = std::min(w, adjacencyMatrix(i, parents[i]));
            }
        }
        return w;
    }

    T WeightOfWeakestEdge()
    {
        T w = std::numeric_limits<T>::max();
        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                w = std::min(w, adjacencyMatrix(i, j));
            }
        }
        return w;
    }


    void CreateDotFile(const std::string& file, const std::vector<std::string>& node_names = {})
    {
        std::ofstream strm(file);
        SAIGA_ASSERT(strm.is_open());


        auto nameforid = [](int id) { return "v_" + std::to_string(id); };

        strm << "graph graphname{" << std::endl;


        for (int i = 0; i < n; ++i)
        {
            auto name = node_names.empty() ? std::to_string(i) : node_names[i];
            strm << "\t" << nameforid(i) << " [label=\"" << name << "\"]" << std::endl;
        }
        strm << std::endl;

        for (int i = 0; i < n; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                auto v = adjacencyMatrix(i, j);
                if (v == invalid_weight) continue;


                strm << "\t" << nameforid(i) << " -- " << nameforid(j) << " [label=\"" << v
                     << "\", dir=none, len=" << 1.0 / (v / 300.0) << "]" << std::endl;
            }
        }

        strm << "}" << std::endl;
    }



    std::vector<Edge> GetEdgesForNode(int node)
    {
        std::vector<WeightedUndirectedGraph::Edge> result;
        for (int i = 0; i < n; ++i)
        {
            auto v = adjacencyMatrix(node, i);
            if (v != invalid_weight)
            {
                Edge e;
                e.from   = node;
                e.to     = i;
                e.weight = v;
                result.push_back(e);
            }
        }
        return result;
    }

    int n;

   private:
    Eigen::Matrix<T, -1, -1> adjacencyMatrix;

    // Tmp variables to store and compute the MST
    std::vector<bool> visited;
    std::vector<int> parents;
    std::vector<T> current_weight;
};
}  // namespace Saiga
