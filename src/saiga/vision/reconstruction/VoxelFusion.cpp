/**
 * Copyright (c) 2021 Darius Rückert
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
static std::stringstream strm;

void FusionScene::Preprocess()
{
    triangle_soup_inclusive_prefix_sum.clear();
    triangle_soup.clear();
    mesh = UnifiedMesh();
    tsdf = std::make_unique<SparseTSDF>(params.voxelSize, params.block_count, params.hash_size);

    if (images.empty()) return;

    depth_map_size = images.front().depthMap.dimensions();
    unproject_undistort_map.create(depth_map_size);

    ProgressBar loading_bar(params.verbose ? std::cout : strm, "Preprocess ", depth_map_size.rows);
#pragma omp parallel for
    for (int i = 0; i < unproject_undistort_map.rows; ++i)
    {
        for (auto j : unproject_undistort_map.colRange())
        {
            Vec2 p(j, i);
            p = K.unproject2(p);
            p = undistortPointGN(p, p, dis);

            unproject_undistort_map(i, j) = p.cast<float>();
        }
        loading_bar.addProgress(1);
    }
}



void FusionScene::AnalyseSparseStructure()
{
    ProgressBar loading_bar(params.verbose ? std::cout : strm, "Analysing  ", Size());

    // #pragma omp parallel for
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
                if (depth <= 0 || depth > params.maxIntegrationDistance) continue;

                float truncation_distance = params.truncationDistance + params.truncationDistanceScale * depth;
                auto truncation_distance2 =
                    std::max(params.min_truncation_factor * params.voxelSize, truncation_distance);



                float min_depth = clamp(depth - truncation_distance2, 0, params.maxIntegrationDistance);
                float max_depth = clamp(depth + truncation_distance2, 0, params.maxIntegrationDistance);


                //                min_depth = depth - 1;
                //                max_depth = depth + 1;

                vec2 p       = unproject_undistort_map(i, j);
                vec3 ray_min = invV * (vec3(p(0), p(1), 1) * min_depth);
                vec3 ray_max = invV * (vec3(p(0), p(1), 1) * max_depth);
                vec3 center  = invV * (vec3(p(0), p(1), 1) * depth);

                if (params.use_confidence)
                {
                    dm.unprojected_position(i, j) = center;
                }

                SAIGA_ASSERT(!params.point_based);
                if (params.point_based)
                {
                    tsdf->AllocateAroundPoint(center);
                    continue;
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
                    tsdf->InsertBlock(idCurrentVoxel);
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


void FusionScene::ComputeWeight()
{
    if (!params.use_confidence)
    {
        return;
    }
    ProgressBar loading_bar(params.verbose ? std::cout : strm, "Comp Weight", Size());
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

                //                if (w < 0.5) w = 0;

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

void FusionScene::Visibility()
{
    {
        ProgressBar loading_bar(params.verbose ? std::cout : strm, "Visibility ", Size());

        auto K2 = K;

        if (params.increase_visibility_frustum)
        {
            K2.fx *= 0.95;
            K2.fy *= 0.95;
        }


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
}


void FusionScene::Integrate()
{
    Visibility();
    {
        ProgressBar loading_bar(params.verbose ? std::cout : strm, "Integrate  ", Size());

        for (int i = 0; i < Size(); ++i)
        {
            auto& dm = images[i];

#pragma omp parallel for
            for (int i = 0; i < (int)dm.visible_blocks.size(); ++i)
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

                            Vec2 np          = pos.head<2>() / pos.z();
                            float voxelDepth = pos.z();

                            np = distortNormalizedPoint(np, dis);


                            Vec2 ip = K.normalizedToImage(np);
                            ip += params.ip_offset;

                            // the voxel is behind the camera
                            if (voxelDepth <= 0) continue;


                            // nearest neighbour lookup
                            Vec2 ip_rounded = ip.array().round();
                            int ipx         = ip_rounded(0);
                            int ipy         = ip_rounded(1);

                            if (dm.depthMap.distanceFromEdge(ipy, ipx) <= 2)
                            {
                                continue;
                            }



                            float imageDepth;

                            if (params.bilinear_intperpolation)
                            {
                                // Bilinear interpolation (reduces artifacts)
                                int ipx = std::floor(ip(0));
                                int ipy = std::floor(ip(1));
                                auto a1 = dm.depthMap(ipy, ipx);
                                auto a4 = dm.depthMap(ipy, ipx + 1);
                                auto a2 = dm.depthMap(ipy + 1, ipx);
                                auto a3 = dm.depthMap(ipy + 1, ipx + 1);
                                if (a1 <= 0 || a2 <= 0 || a3 <= 0 || a4 <= 0) continue;
                                imageDepth = dm.depthMap.inter(ip(1), ip(0));
                            }
                            else
                            {
                                // SAIGA_EXIT_ERROR("unimplemented");
                                imageDepth = dm.depthMap(ipy, ipx);
                                if (imageDepth <= 0) continue;
                            }

                            // No valid depth
                            if (imageDepth <= 0) continue;
                            if (imageDepth > params.maxIntegrationDistance) continue;
                            float confidence = params.use_confidence ? dm.confidence(ipy, ipx) : 1;
                            if (confidence <= 0) continue;

                            // current td
                            float truncation_distance =
                                params.truncationDistance + params.truncationDistanceScale * imageDepth;
                            truncation_distance =
                                std::max(params.min_truncation_factor * params.voxelSize, truncation_distance);


                            float new_tsdf       = imageDepth - voxelDepth;
                            auto new_weight      = params.newWeight * confidence;
                            float current_tsdf   = cell.distance;
                            float current_weight = cell.weight;


                            if (params.ground_truth_fuse)
                            {
                                // A fusion algorithm which assumes perfect input data.
                                // Therefore we don't need to average the distance results
                                // It is enough to use the minimum observation
                                if (new_tsdf < -truncation_distance * params.ground_truth_trunc_factor)
                                {
                                    continue;
                                }


                                new_tsdf = clamp(new_tsdf, -params.sd_clamp, params.sd_clamp);



                                if (current_weight == 0)
                                {
                                    cell.distance = new_tsdf;
                                    cell.weight   = new_weight;
                                }


                                if (current_tsdf < 0 && new_tsdf > 0)
                                {
                                    cell.distance = new_tsdf;
                                    cell.weight   = new_weight;
                                }

                                if (current_tsdf < 0 && new_tsdf < 0)
                                {
                                    cell.distance = std::max(current_tsdf, new_tsdf);
                                    cell.weight   = std::min(params.maxWeight, current_weight + new_weight);
                                }

                                if (current_tsdf > 0 && new_tsdf > 0)
                                {
                                    cell.distance = std::min(current_tsdf, new_tsdf);
                                    cell.weight   = std::min(params.maxWeight, current_weight + new_weight);
                                }

                                if (current_tsdf > 0 && new_tsdf < 0)
                                {
                                    // do nothing
                                }

                                continue;
                            }



                            if (new_tsdf < -truncation_distance)
                            {
                                continue;
                            }


                            new_tsdf = clamp(new_tsdf, -params.sd_clamp, params.sd_clamp);



                            if (current_weight == 0)
                            {
                                cell.distance = new_tsdf;
                                cell.weight   = new_weight;
                            }
                            else
                            {
#if 0
                                float updated_tsdf;
                                if (std::abs(current_tsdf - new_tsdf) < params.max_distance_error)
                                {
                                    updated_tsdf = (current_weight * current_tsdf + add_weight * new_tsdf) /
                                                   (current_weight + add_weight);
                                }
                                else
                                {
                                    if (std::abs(current_tsdf) < std::abs(new_tsdf))
                                    {
                                        updated_tsdf = current_tsdf;
                                    }
                                    else
                                    {
                                        updated_tsdf = new_tsdf;
                                    }
                                }
#else

                                float updated_tsdf = (current_weight * current_tsdf + new_weight * new_tsdf) /
                                                     (current_weight + new_weight);
#endif

                                float updated_weight = std::min(params.maxWeight, current_weight + new_weight);
                                cell.distance        = updated_tsdf;
                                cell.weight          = updated_weight;
                            }
                        }
                    }
                }
            }

            loading_bar.addProgress(1);
        }
    }
}

void FusionScene::IntegratePointBased()
{
    Visibility();
    tsdf->SetForAll(500, 0);

    {
        ProgressBar loading_bar(params.verbose ? std::cout : strm, "IntegrateP ", Size());

        for (int i = 0; i < Size(); ++i)
        {
            auto& dm = images[i];

            for (int i = 0; i < (int)dm.visible_blocks.size(); ++i)

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
                            Vec2 ip_rounded = ip.array().round();
                            int ipx         = ip_rounded(0);
                            int ipy         = ip_rounded(1);

                            if (dm.depthMap.distanceFromEdge(ipy, ipx) <= 2)
                            {
                                continue;
                            }

                            float imageDepth;
                            {
                                int ipx = std::floor(ip(0));
                                int ipy = std::floor(ip(1));
                                auto a1 = dm.depthMap(ipy, ipx);
                                auto a2 = dm.depthMap(ipy + 1, ipx);
                                auto a3 = dm.depthMap(ipy + 1, ipx + 1);
                                auto a4 = dm.depthMap(ipy, ipx + 1);
                                if (a1 <= 0 || a2 <= 0 || a3 <= 0 || a4 <= 0) continue;
                            }
                            imageDepth = dm.depthMap.inter(ip(1), ip(0));


                            int r = 1;

                            float min_dis = 1000;
                            //                            float positive = 0;

                            for (int y = -r; y <= r; ++y)
                            {
                                for (int x = -r; x <= r; ++x)
                                {
                                    if (dm.depthMap(ipy + y, ipx + x) <= 0) continue;
                                    vec3 p  = dm.V.cast<float>() * dm.unprojected_position(ipy + y, ipx + x);
                                    float d = (pos.cast<float>() - p).norm();
                                    min_dis = std::min(min_dis, d);

                                    // positive += (p.z() > voxelDepth) ? 1 : -1;
                                }
                            }

                            // if (min_dis > params.truncationDistance) continue;


                            //                            if (positive > 0)
                            //                                positive = 1;
                            //                            else if (positive < 0)
                            //                                positive = -1;
                            //                            else
                            //                                continue;

                            double surface_distance = std::abs(imageDepth - voxelDepth);

                            if (min_dis < cell.distance)
                            {
                                cell.distance = surface_distance;
                                cell.weight   = (voxelDepth < imageDepth) ? 1 : -1;
                                continue;
                            }
                            cell.distance = std::min(cell.distance, min_dis);

                            if (surface_distance < -params.truncationDistance)
                            {
                                continue;
                            }

                            if (cell.distance + params.truncationDistance < min_dis)
                            {
                                //                                continue;
                            }

                            //                            cell.weight += 1;  // positive;
                            cell.weight += (voxelDepth < imageDepth) ? 5 : -1;
                        }
                    }
                }
            }
            loading_bar.addProgress(1);
        }
    }


    for (int b = 0; b < tsdf->current_blocks; ++b)
    {
        auto& block = tsdf->blocks[b];

        for (auto& z : block.data)
            for (auto& y : z)
                for (auto& x : y)
                {
                    if (x.weight < 0)
                    {
                        x.distance = -x.distance;
                        x.weight   = -x.weight;
                    }
                }
    }



#if 0
    for (int i = 0; i < Size(); ++i)
    {
        auto& dm  = images[i];
        auto invV = dm.V.inverse().cast<float>();

        for (int i = 0; i < dm.depthMap.rows; ++i)
        {
            for (auto j : dm.depthMap.colRange())
            {
                auto depth = dm.depthMap(i, j);

                if (depth <= 0) continue;
                vec2 p      = unproject_undistort_map(i, j);
                vec3 center = invV * (vec3(p(0), p(1), 1) * depth);


                auto block_id = tsdf->GetBlockIndex(center);

                int r = 1;

                for (int z = -r; z <= r; ++z)
                {
                    for (int y = -r; y <= r; ++y)
                    {
                        for (int x = -r; x <= r; ++x)
                        {
                            ivec3 current_id = ivec3(x, y, z) + block_id;
                            auto b           = tsdf->GetBlock(current_id);
                            //                            tsdf->GlobalBlockOffset()
                            SAIGA_ASSERT(b);

                            for (int i = 0; i < tsdf->VOXEL_BLOCK_SIZE; ++i)
                            {
                                for (int j = 0; j < tsdf->VOXEL_BLOCK_SIZE; ++j)
                                {
                                    for (int k = 0; k < tsdf->VOXEL_BLOCK_SIZE; ++k)
                                    {
                                        vec3 global_pos           = tsdf->GlobalPosition(current_id, i, j, k);
                                        float dis                 = (global_pos - center).norm();
                                        b->data[i][j][k].distance = std::min(dis, b->data[i][j][k].distance);
                                        b->data[i][j][k].weight   = 1;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
#endif
}

void FusionScene::ExtractMesh()
{
    mesh = UnifiedMesh();

    auto triangle_soup_per_block =
        tsdf->ExtractSurface(params.extract_iso, params.extract_outlier_factor, 0, 4, params.verbose);

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

    mesh = tsdf->CreateMesh(triangle_soup_per_block, params.post_process_mesh);
    //    for (auto& t : triangle_soup)
    //    {
    //        VertexNC tri[3];
    //        for (int i = 0; i < 3; ++i)
    //        {
    //            tri[i].position.head<3>() = t[i].cast<float>();
    //            tri[i].color              = vec4(1, 1, 1, 1);
    //        }
    //        mesh.addTriangle(tri);
    //    }

    //    {
    //        mesh.sortVerticesByPosition();
    //        mesh.removeSubsequentDuplicates();
    //        mesh.removeDegenerateFaces();
    //        mesh.computePerVertexNormal();
    //    }


    if (!params.out_file.empty())
    {
        std::ofstream strm(params.out_file);
        saveMeshOff(mesh.Mesh<VertexNC,uint32_t>(), strm);
    }
}



void FusionScene::Fuse()
{
    std::cout << "Fusing " << Size() << " depth maps..." << std::endl;
    Preprocess();
    AnalyseSparseStructure();
    ComputeWeight();
    if (params.point_based)
    {
        IntegratePointBased();
    }
    else
    {
        Integrate();
    }
    ExtractMesh();
    std::cout << *tsdf << std::endl;
}


void FusionScene::FuseIncrement(const FusionImage& image, bool first)
{
    images.clear();
    images.push_back(image);

    if (first)
    {
        Preprocess();
    }
    AnalyseSparseStructure();
    ComputeWeight();
    Integrate();
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
    ImGui::InputFloat("min_truncation_factor", &min_truncation_factor);
    ImGui::Checkbox("use_confidence", &use_confidence);
    ImGui::Checkbox("bilinear_intperpolation", &bilinear_intperpolation);

    ImGui::Checkbox("test", &test);

    static std::string buffer;
    ImGui::InputText("Out File", &buffer);
    out_file = buffer;
}



}  // namespace Saiga
