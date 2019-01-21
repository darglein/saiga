/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PoseGraph.h"

#include "saiga/vision/Random.h"

#include <fstream>
namespace Saiga
{
void PoseGraph::addNoise(double stddev)
{
    for (auto& e : poses)
    {
        e.translation() += Random::gaussRandMatrix<Vec3>(0, stddev);
    }
}

Vec6 PoseGraph::residual6(const PoseEdge& edge)
{
    auto& _from = poses[edge.from];
    auto& _to   = poses[edge.to];
#ifdef LSD_REL
    auto error_ = _from.inverse() * _to * edge.meassurement.inverse();
#else
    auto error_ = edge.meassurement.inverse() * _to * _from.inverse();
#endif
    return error_.log() * edge.weight;
}

double PoseGraph::chi2()
{
    double error = 0;
    for (PoseEdge& e : edges)
    {
        double sqerror;
        sqerror = residual6(e).squaredNorm();
        error += sqerror;
    }
    return error;
}

void PoseGraph::save(const std::string& file)
{
    cout << "Saving PoseGraph to " << file << "." << endl;
    cout << "chi2 " << chi2() << endl;
    std::ofstream strm(file);
    SAIGA_ASSERT(strm.is_open());
    strm.precision(20);
    strm << std::scientific;

    strm << poses.size() << " " << edges.size() << endl;
    for (auto& e : poses)
    {
        strm << e.params().transpose() << endl;
    }
    for (auto& e : edges)
    {
        strm << e.from << " " << e.to << " " << e.weight << " " << e.meassurement.params().transpose() << endl;
    }
}

void PoseGraph::load(const std::string& file)
{
    cout << "Loading scene from " << file << "." << endl;


    std::ifstream strm(file);
    SAIGA_ASSERT(strm.is_open());


    auto consumeComment = [&]() {
        while (true)
        {
            auto c = strm.peek();
            if (c == '#')
            {
                std::string s;
                std::getline(strm, s);
            }
            else
            {
                break;
            }
        }
    };


    consumeComment();
    int num_vertices, num_edges;
    strm >> num_vertices >> num_edges;
    poses.resize(num_vertices);
    edges.resize(num_edges);
    cout << "Vertices/Edges: " << num_vertices << "/" << num_edges << endl;

    for (auto& e : poses)
    {
        Eigen::Map<Sophus::Vector<double, SE3::num_parameters>> v2(e.data());
        Sophus::Vector<double, SE3::num_parameters> v;
        strm >> v;
        v2 = v;
    }

    for (auto& e : edges)
    {
        Eigen::Map<Sophus::Vector<double, SE3::num_parameters>> v2(e.meassurement.data());
        Sophus::Vector<double, SE3::num_parameters> v;
        strm >> e.from >> e.to >> e.weight >> v;
        v2 = v;

        e.setRel(poses[e.from], poses[e.to]);
    }
}

}  // namespace Saiga
