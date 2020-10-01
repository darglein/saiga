/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "VoxelFusion.h"

#include "saiga/core/geometry/all.h"
#include "saiga/core/imgui/imgui.h"

#include "MarchingCubes.h"
#include "fstream"

namespace Saiga
{
void FusionScene::Preprocess()
{
    if (images.empty()) return;

    depth_map_size = images.front().depthMap.dimensions();

    unproject_undistort_map.create(depth_map_size);


    for (auto i : unproject_undistort_map.rowRange())
    {
        for (auto j : unproject_undistort_map.colRange())
        {
            Vec2 p(j, i);
            p = K.unproject2(p);
            p = undistortPointGN(p, p, dis);

            unproject_undistort_map(i, j) = p.cast<float>();
        }
    }

    tsdf = std::make_unique<SparseTSDF<8>>(params.voxelSize, 250 * 1000, 5 * 1000 * 1000);
}

void FusionScene::AnalyseSparseStructure()
{
    SyncedConsoleProgressBar loading_bar(std::cout, "Analysing Structure", Size());


#pragma omp parallel for
    for (int i = 0; i < Size(); ++i)
    {
        auto& dm  = images[i];
        auto invV = dm.V.inverse().cast<float>();

        if (params.use_confidence)
        {
            dm.unprojected_position.create(dm.depthMap.dimensions());
            dm.unprojected_position.makeZero();
        }

        //        std::set<std::tuple<int, int, int>> leset;

        //        for (auto i : dm.depthMap.rowRange())
        for (int i = 0; i < dm.depthMap.rows; ++i)
        {
            for (auto j : dm.depthMap.colRange())
            {
                auto depth = dm.depthMap(i, j);

                float truncation_distance = params.truncationDistance + params.truncationDistanceScale * depth;
                truncation_distance = std::max(params.min_truncation_factor * params.voxelSize, truncation_distance);

                if (depth <= 0 || depth > params.maxIntegrationDistance) continue;


                float min_depth = clamp(depth - truncation_distance, 0, params.maxIntegrationDistance);
                float max_depth = clamp(depth + truncation_distance, 0, params.maxIntegrationDistance);


                //                min_depth = depth - 1;
                //                max_depth = depth + 1;

                vec2 p       = unproject_undistort_map(i, j);
                vec3 ray_min = invV * (vec3(p(0), p(1), 1) * min_depth);
                vec3 ray_max = invV * (vec3(p(0), p(1), 1) * max_depth);

                if (params.use_confidence)
                {
                    vec3 center                   = invV * (vec3(p(0), p(1), 1) * depth);
                    dm.unprojected_position(i, j) = center;
                }

                if (min_depth >= max_depth) continue;

                vec3 rayDir = (ray_max - ray_min).normalized();

                auto idCurrentVoxel = tsdf->GetBlockIndex(ray_min);
                auto idEnd          = tsdf->GetBlockIndex(ray_max);

                vec3 step        = rayDir.array().sign();
                vec3 boundaryPos = tsdf->GlobalBlockOffset(
                                       idCurrentVoxel + step.cast<int>().array().max(ivec3::Zero().array()).matrix()) -
                                   make_vec3(0.5f * params.voxelSize);
                vec3 tMax   = (boundaryPos - ray_min).array() / rayDir.array();
                vec3 tDelta = (step * tsdf->VOXEL_BLOCK_SIZE * params.voxelSize).array() / rayDir.array();


                ivec3 idBound = idEnd + step.cast<int>();


                auto inf = std::numeric_limits<float>::infinity();
                if (rayDir.x() == 0.0f)
                {
                    tMax.x()   = inf;
                    tDelta.x() = inf;
                }
                if (rayDir.y() == 0.0f)
                {
                    tMax.y()   = inf;
                    tDelta.y() = inf;
                }
                if (rayDir.z() == 0.0f)
                {
                    tMax.z()   = inf;
                    tDelta.z() = inf;
                }


                if (boundaryPos.x() - ray_min.x() == 0.0f)
                {
                    tMax.x()   = inf;
                    tDelta.x() = inf;
                }
                if (boundaryPos.y() - ray_min.y() == 0.0f)
                {
                    tMax.y()   = inf;
                    tDelta.y() = inf;
                }
                if (boundaryPos.z() - ray_min.z() == 0.0f)
                {
                    tMax.z()   = inf;
                    tDelta.z() = inf;
                }

                while (true)
                {
                    //                    leset.insert({idCurrentVoxel(0), idCurrentVoxel(1), idCurrentVoxel(2)});
                    tsdf->InsertBlockLock(idCurrentVoxel);
                    // Traverse voxel grid
                    if (tMax.x() < tMax.y() && tMax.x() < tMax.z())
                    {
                        idCurrentVoxel.x() += step.x();
                        if (idCurrentVoxel.x() == idBound.x()) break;
                        tMax.x() += tDelta.x();
                    }
                    else if (tMax.z() < tMax.y())
                    {
                        idCurrentVoxel.z() += step.z();
                        if (idCurrentVoxel.z() == idBound.z()) break;
                        tMax.z() += tDelta.z();
                    }
                    else
                    {
                        idCurrentVoxel.y() += step.y();
                        if (idCurrentVoxel.y() == idBound.y()) break;
                        tMax.y() += tDelta.y();
                    }
                }
            }
        }

        //        dm.truncated_blocks.clear();
        //        for (auto i : leset)
        //        {
        //            dm.truncated_blocks.push_back({std::get<0>(i), std::get<1>(i), std::get<2>(i)});
        //        }

        loading_bar.addProgress(1);
    }
}

void FusionScene::Allocate()
{
    //    return;
    //    SyncedConsoleProgressBar loadingBar(std::cout, "Allocating", Size());

    //#pragma omp parallel for
    //    for (int i = 0; i < Size(); ++i)
    //    {
    //        auto& dm = images[i];
    //        for (auto i : dm.truncated_blocks)
    //        {
    //            tsdf->InsertBlockLock(i);
    //        }
    //        loadingBar.addProgress(1);

    //        dm.truncated_blocks = {};
    //    }
}


void FusionScene::ComputeWeight()
{
    if (!params.use_confidence)
    {
        return;
    }
    SyncedConsoleProgressBar loading_bar(std::cout, "ComputeWeight", Size());
#pragma omp parallel for
    for (int i = 0; i < Size(); ++i)
    {
        auto& dm = images[i];

        dm.confidence.create(dm.depthMap.dimensions());
        dm.confidence.makeZero();

        vec3 eye = dm.V.inverse().translation().cast<float>();

        for (auto i : dm.depthMap.rowRange(1))
        {
            for (auto j : dm.depthMap.colRange(1))
            {
                auto depth = dm.depthMap(i, j);

                if (depth <= 0 || dm.depthMap(i + 1, j) <= 0 || dm.depthMap(i - 1, j) <= 0 ||
                    dm.depthMap(i, j + 1) <= 0 || dm.depthMap(i, j - 1) <= 0)
                {
                    continue;
                }

                vec3& c = dm.unprojected_position(i, j);

                vec3 v = (eye - c).normalized();

                vec3& l = dm.unprojected_position(i - 1, j);
                vec3& r = dm.unprojected_position(i + 1, j);

                vec3& d = dm.unprojected_position(i, j - 1);
                vec3& u = dm.unprojected_position(i, j + 1);

                vec3 n = (l - r).cross(d - u).normalized();


                float w = v.dot(n);
                w       = clamp(w, 0, 1);

                float wd =
                    1.0 / (depth + 1);  // (params.maxIntegrationDistance - depth) / params.maxIntegrationDistance;
                wd = clamp(wd, 0, 1);

                dm.confidence(i, j) = w * wd;
                // std::cout << "w: " << w << " " << depth << " -> " << wd << std::endl;
            }
        }
        //        exit(0);
        loading_bar.addProgress(1);
    }
}


void FusionScene::Integrate()
{
#if 0
    {
        SyncedConsoleProgressBar loading_bar(std::cout, "Integrate", Size());

        auto K2 = K;
        K2.fx *= 0.95;
        K2.fy *= 0.95;


        for (int i = 0; i < Size(); ++i)
        {
            auto& dm = images[i];
            dm.visible_blocks.clear();


#    pragma omp parallel for
            for (int i = 0; i < tsdf->current_blocks; ++i)
            {
                auto& block = tsdf->blocks[i];
                auto id     = block.index;
                Vec3 c      = tsdf->BlockCenter(block.index).cast<double>();

                // project to image
                Vec3 pos = dm.V * c;

                Vec2 np           = pos.head<2>() / pos.z();
                double voxelDepth = pos.z();

                if (voxelDepth < 0 || voxelDepth > params.maxIntegrationDistance + 0.4) continue;

                np      = distortNormalizedPoint(np, dis);
                Vec2 ip = K2.normalizedToImage(np);

                // nearest neighbour lookup
                ip      = ip.array().round();
                int ipx = ip(0);
                int ipy = ip(1);

                if (!dm.depthMap.inImage(ipy, ipx))
                {
                    continue;
                }

                //                dm.visible_blocks.push_back(block.index);

                for (int i = 0; i < tsdf->VOXEL_BLOCK_SIZE; ++i)
                {
                    for (int j = 0; j < tsdf->VOXEL_BLOCK_SIZE; ++j)
                    {
                        for (int k = 0; k < tsdf->VOXEL_BLOCK_SIZE; ++k)
                        {
                            Vec3 global_pos = tsdf->GlobalPosition(id, i, j, k).cast<double>();
                            auto& cell      = block.data[i][j][k];



                            // project to image
                            Vec3 pos = dm.V * global_pos;

                            Vec2 np           = pos.head<2>() / pos.z();
                            double voxelDepth = pos.z();

                            np = distortNormalizedPoint(np, dis);


                            Vec2 ip = K.normalizedToImage(np);

                            // the voxel is behind the camera
                            if (voxelDepth <= 0) continue;

                            // nearest neighbour lookup
                            ip      = ip.array().round();
                            int ipx = ip(0);
                            int ipy = ip(1);

                            if (!dm.depthMap.inImage(ipy, ipx))
                            {
                                continue;
                            }

                            auto imageDepth = dm.depthMap(ipy, ipx);
                            if (imageDepth <= 0) continue;

                            if (imageDepth > params.maxIntegrationDistance) continue;



                            float confidence = params.use_confidence ? dm.confidence(ipy, ipx) : 1;

                            if (confidence <= 0) continue;

                            double surface_distance = imageDepth - voxelDepth;

                            float truncation_distance =
                                params.truncationDistance + params.truncationDistanceScale * imageDepth;
                            truncation_distance =
                                std::max(params.min_truncation_factor * params.voxelSize, truncation_distance);

                            if (surface_distance >= -truncation_distance)
                            {
                                auto new_tsdf       = surface_distance;
                                auto current_tsdf   = cell.distance;
                                auto current_weight = cell.weight;

                                auto add_weight = params.newWeight * confidence;

                                if (current_weight == 0)
                                {
                                    cell.distance = new_tsdf;
                                    cell.weight   = add_weight;
                                }
                                else
                                {
                                    double updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                          (current_weight + add_weight);

                                    auto new_weight = current_weight + add_weight;

                                    new_weight = std::min(params.maxWeight, new_weight);

                                    cell.distance = updated_tsdf;
                                    cell.weight   = new_weight;
                                }
                            }
                        }
                    }
                }
            }
            loading_bar.addProgress(1);
        }
    }
    return;
#endif

    {
        SyncedConsoleProgressBar loading_bar(std::cout, "Visibility", Size());

        auto K2 = K;
        K2.fx *= 0.95;
        K2.fy *= 0.95;


#pragma omp parallel for
        for (int i = 0; i < Size(); ++i)
        {
            auto& dm = images[i];
            dm.visible_blocks.clear();


            //            for (auto& block : tsdf->blocks)
            for (int i = 0; i < tsdf->current_blocks; ++i)
            {
                auto block = tsdf->blocks[i];
                Vec3 c     = tsdf->BlockCenter(block.index).cast<double>();

                // project to image
                Vec3 pos = dm.V * c;

                Vec2 np           = pos.head<2>() / pos.z();
                double voxelDepth = pos.z();

                if (voxelDepth < 0 || voxelDepth > params.maxIntegrationDistance + 0.4) continue;

                np      = distortNormalizedPoint(np, dis);
                Vec2 ip = K2.normalizedToImage(np);

                // nearest neighbour lookup
                ip      = ip.array().round();
                int ipx = ip(0);
                int ipy = ip(1);

                if (!dm.depthMap.inImage(ipy, ipx))
                {
                    continue;
                }

                dm.visible_blocks.push_back(block.index);
            }
            loading_bar.addProgress(1);
        }
    }

    {
        SyncedConsoleProgressBar loading_bar(std::cout, "Integrate", Size());

        for (int i = 0; i < Size(); ++i)
        {
            auto& dm = images[i];

#pragma omp parallel for
            for (int i = 0; i < dm.visible_blocks.size(); ++i)
            {
                auto& id    = dm.visible_blocks[i];
                auto* block = tsdf->GetBlock(id);
                SAIGA_ASSERT(block);
                SAIGA_ASSERT(block->index == id);

                //        Vec3 offset = tsdf.GlobalBlockOffset(id).cast<double>();

                for (int i = 0; i < tsdf->VOXEL_BLOCK_SIZE; ++i)
                {
                    for (int j = 0; j < tsdf->VOXEL_BLOCK_SIZE; ++j)
                    {
                        for (int k = 0; k < tsdf->VOXEL_BLOCK_SIZE; ++k)
                        {
                            Vec3 global_pos = tsdf->GlobalPosition(id, i, j, k).cast<double>();
                            auto& cell      = block->data[i][j][k];



                            // project to image
                            Vec3 pos = dm.V * global_pos;

                            Vec2 np           = pos.head<2>() / pos.z();
                            double voxelDepth = pos.z();

                            np = distortNormalizedPoint(np, dis);


                            Vec2 ip = K.normalizedToImage(np);

                            // the voxel is behind the camera
                            if (voxelDepth <= 0) continue;

                            // nearest neighbour lookup
                            ip      = ip.array().round();
                            int ipx = ip(0);
                            int ipy = ip(1);

                            if (!dm.depthMap.inImage(ipy, ipx))
                            {
                                continue;
                            }

                            auto imageDepth = dm.depthMap(ipy, ipx);
                            if (imageDepth <= 0) continue;

                            if (imageDepth > params.maxIntegrationDistance) continue;



                            float confidence = params.use_confidence ? dm.confidence(ipy, ipx) : 1;

                            if (confidence <= 0) continue;

                            //                    Vec3 ipUnproj = dm->K.unproject(ip, imageDepth);

                            //                    double lambda = Vec3(pos(0), pos(1), 1).norm();
                            //                    double sdf    = (-1.f) * ((1.f / lambda) * pos.norm() - imageDepth);
                            //                    double sdf = (ipUnproj - pos).norm();
                            double surface_distance = imageDepth - voxelDepth;

                            //                    if (imageDepth < voxelDepth) sdf *= -1;



                            float truncation_distance =
                                params.truncationDistance + params.truncationDistanceScale * imageDepth;
                            truncation_distance =
                                std::max(params.min_truncation_factor * params.voxelSize, truncation_distance);

                            //                    if ( std::abs(sdf) < -truncation_distance)
                            if (surface_distance >= -truncation_distance)
                            {
                                //                        sdf = std::clamp(sdf,-truncation_distance)
                                //                                auto new_tsdf       = std::min(1., sdf /
                                //                                truncation_distance);
                                auto new_tsdf       = surface_distance;  // std::min(1., sdf);
                                auto current_tsdf   = cell.distance;
                                auto current_weight = cell.weight;

                                auto add_weight = params.newWeight * confidence;

                                if (current_weight == 0)
                                {
                                    cell.distance = new_tsdf;
                                    cell.weight   = add_weight;
                                }
                                else
                                {
                                    double updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                          (current_weight + add_weight);

                                    auto new_weight = current_weight + add_weight;

                                    new_weight = std::min(params.maxWeight, new_weight);

                                    cell.distance = updated_tsdf;
                                    cell.weight   = new_weight;
                                }
                            }
                        }
                    }
                }
            }

            loading_bar.addProgress(1);
        }
    }
}

void FusionScene::ExtractMesh()
{
    mesh.clear();

    auto triangle_soup_per_block = tsdf->ExtractSurface(0);

    int sum = 0;
    for (auto& v : triangle_soup_per_block)
    {
        for (auto& t : v)
        {
            triangle_soup.push_back(t);
        }
        sum += v.size();
        triangle_soup_inclusive_prefix_sum.push_back(sum);
    }

    for (auto& t : triangle_soup)
    {
        VertexNC tri[3];
        for (int i = 0; i < 3; ++i)
        {
            tri[i].position.head<3>() = t[i].cast<float>();
            tri[i].color              = vec4(1, 1, 1, 1);
        }
        mesh.addTriangle(tri);
    }

    {
        mesh.sortVerticesByPosition();
        mesh.removeSubsequentDuplicates();
        mesh.removeDegenerateFaces();
        mesh.computePerVertexNormal();
    }


    {
        std::ofstream strm(params.out_file);
        mesh.saveMeshOff(strm);
        std::cout << mesh << " saved as " << params.out_file << std::endl;
    }
}



void FusionScene::Fuse()
{
    std::cout << "Fusing " << Size() << " depth maps..." << std::endl;
    Preprocess();
    AnalyseSparseStructure();
    Allocate();
    ComputeWeight();
    Integrate();
    ExtractMesh();
    tsdf->Print(std::cout);
}



void FusionScene::imgui()
{
    params.imgui();

    //    if (ImGui::Button("Fuse Depth Maps"))
    //    {
    //        Fuse();
    //    }
}

void FusionParams::imgui()
{
    ImGui::InputFloat("voxelSize", &voxelSize);
    ImGui::InputFloat("truncationDistance", &truncationDistance);
    ImGui::InputFloat("truncationDistanceScale", &truncationDistanceScale);
    ImGui::InputFloat("maxIntegrationDistance", &maxIntegrationDistance);
    ImGui::InputFloat("newWeight", &newWeight);
    ImGui::InputFloat("maxWeight", &maxWeight);
    ImGui::Checkbox("use_confidence", &use_confidence);
}



}  // namespace Saiga
