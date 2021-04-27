/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "SparseTSDF.h"

#include "saiga/core/util/file.h"
#include "saiga/core/util/zlib.h"
namespace Saiga
{
void SparseTSDF::EraseEmptyBlocks()
{
    for (int i = 0; i < current_blocks; ++i)
    {
        auto& b = blocks[i];
        if (b.Empty())
        {
            // std::cout << "erase empty " << b.index.transpose() << std::endl;
            EraseBlock(b.index);
            i--;
        }
    }
}

std::vector<std::vector<SparseTSDF::Triangle>> SparseTSDF::ExtractSurface(double iso, float outlier_factor,
                                                                          float min_weight, int threads, bool verbose)
{
    std::stringstream sstrm;
    ProgressBar loading_bar(verbose ? std::cout : sstrm, "Ex. Surface", current_blocks);

    //        std::vector<std::vector<std::array<vec3, 3>>> triangle_soup_thread(threads);

    // Each block generates a list of triangles
    std::vector<std::vector<Triangle>> triangle_soup_per_block(current_blocks);

#pragma omp parallel for num_threads(threads)
    for (int b = 0; b < current_blocks; ++b)
    {
        auto& triangle_soup = triangle_soup_per_block[b];
        auto& block         = blocks[b];
        // Compute positions and values of (n+1) x (n+1) x (n+1) block.
        // The (+1) data point is taken from neighbouring blocks to close the holes.
        std::pair<vec3, float> local_data[VOXEL_BLOCK_SIZE + 1][VOXEL_BLOCK_SIZE + 1][VOXEL_BLOCK_SIZE + 1];

        // Fill from own block
        for (int i = 0; i < VOXEL_BLOCK_SIZE + 1; ++i)
        {
            for (int j = 0; j < VOXEL_BLOCK_SIZE + 1; ++j)
            {
                for (int k = 0; k < VOXEL_BLOCK_SIZE + 1; ++k)
                {
                    int li = i % VOXEL_BLOCK_SIZE;
                    int lj = j % VOXEL_BLOCK_SIZE;
                    int lk = k % VOXEL_BLOCK_SIZE;

                    int bi = i / VOXEL_BLOCK_SIZE;
                    int bj = j / VOXEL_BLOCK_SIZE;
                    int bk = k / VOXEL_BLOCK_SIZE;

                    VoxelBlockIndex read_block_id = block.index + ivec3(bk, bj, bi);

                    auto* read_block = GetBlock(read_block_id);


                    vec3 p = GlobalPosition(block.index, i, j, k);

                    if (read_block)
                    {
                        float dis           = read_block->data[li][lj][lk].distance;
                        float wei           = read_block->data[li][lj][lk].weight;
                        local_data[i][j][k] = {p, wei > min_weight ? dis : std::numeric_limits<float>::infinity()};
                        //                        local_data[i][j][k] = {p, dis};
                    }
                    else
                    {
                        local_data[i][j][k] = {p, std::numeric_limits<float>::infinity()};
                    }
                }
            }
        }


        // create triangles
        for (int i = 0; i < VOXEL_BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < VOXEL_BLOCK_SIZE; ++j)
            {
                for (int k = 0; k < VOXEL_BLOCK_SIZE; ++k)
                {
                    std::array<std::pair<vec3, float>, 8> cell;

                    cell[0] = local_data[i][j][k];
                    cell[1] = local_data[i][j][k + 1];
                    cell[2] = local_data[i + 1][j][k + 1];
                    cell[3] = local_data[i + 1][j][k];
                    cell[4] = local_data[i][j + 1][k];
                    cell[5] = local_data[i][j + 1][k + 1];
                    cell[6] = local_data[i + 1][j + 1][k + 1];
                    cell[7] = local_data[i + 1][j + 1][k];

                    bool finite   = true;
                    float abs_max = 0;

                    for (auto i = 0; i < 8; ++i)
                    {
                        finite &= std::isfinite(cell[i].second);
                        abs_max = std::max(abs_max, std::abs(cell[i].second));
                    }

                    if (abs_max > outlier_factor * voxel_size)
                    {
                        continue;
                    }

                    if (!finite)
                    {
                        continue;
                    }

                    auto [triangles, count] = MarchingCubes(cell, iso);


                    for (int n = 0; n < count; ++n)
                    {
                        auto tri = triangles[n];
                        triangle_soup.push_back(tri);
                    }
                }
            }
        }
        loading_bar.addProgress(1);
    }


    return triangle_soup_per_block;
}

UnifiedMesh SparseTSDF::CreateMesh(const std::vector<std::vector<SparseTSDF::Triangle>>& triangles, bool post_process)
{
    UnifiedMesh mesh;

    for (auto& v : triangles)
    {
        for (auto& t : v)
        {
            int n = mesh.NumVertices();
            for (int i = 0; i < 3; ++i)
            {
                mesh.position.push_back(t[i].cast<float>());
            }
            mesh.triangles.push_back(ivec3(n, n + 1, n + 2));
        }
    }

    mesh.CalculateVertexNormals();
    mesh.SetVertexColor(vec4(1, 1, 1, 1));

    return mesh;
}


void SparseTSDF::Save(const std::string& file)
{
    BinaryFile strm(file, std::ios_base::out);
    strm << voxel_size << voxel_size_inv << block_size_inv << hash_size << current_blocks;
    strm << blocks;
    strm << first_hashed_block;
}

void SparseTSDF::Load(const std::string& file)
{
    BinaryFile strm(file, std::ios_base::in);
    SAIGA_ASSERT(strm.strm.is_open());
    strm >> voxel_size >> voxel_size_inv >> block_size_inv >> hash_size >> current_blocks;
    strm >> blocks;
    strm >> first_hashed_block;
}

void SparseTSDF::SaveCompressed(const std::string& file)
{
#ifdef SAIGA_USE_ZLIB
    BinaryOutputVector strm;
    strm << voxel_size << voxel_size_inv << block_size_inv << hash_size << current_blocks;
    strm << blocks;
    strm << first_hashed_block;
    auto compressed = compress(strm.data.data(), strm.data.size());
    File::saveFileBinary(file, compressed.data(), compressed.size());
#else
    SAIGA_EXIT_ERROR("zlib not found.");
#endif
}

void SparseTSDF::LoadCompressed(const std::string& file)
{
#ifdef SAIGA_USE_ZLIB
    auto compressed_data = File::loadFileBinary(file);
    auto data            = uncompress(compressed_data.data());
    BinaryInputVector strm(data.data(), data.size());
    strm >> voxel_size >> voxel_size_inv >> block_size_inv >> hash_size >> current_blocks;
    strm >> blocks;
    strm >> first_hashed_block;
#else
    SAIGA_EXIT_ERROR("zlib not found.");
#endif
}

bool SparseTSDF::operator==(const SparseTSDF& other) const
{
    if (voxel_size != other.voxel_size || voxel_size_inv != other.voxel_size_inv ||
        block_size_inv != other.block_size_inv || hash_size != other.hash_size ||
        current_blocks != other.current_blocks || first_hashed_block != other.first_hashed_block)
    {
        return false;
    }

    if (blocks.size() != other.blocks.size()) return false;

    for (int i = 0; i < (int)std::min(blocks.size(), other.blocks.size()); ++i)
    {
        auto& b1 = blocks[i];
        auto& b2 = other.blocks[i];

        auto a1 = (std::array<char, sizeof(VoxelBlock)>*)&b1;
        auto a2 = (std::array<char, sizeof(VoxelBlock)>*)&b2;

        if (*a1 != *a2) return false;
    }

    return true;
}


void SparseTSDF::ClampDistance(float distance)
{
    for (int b = 0; b < current_blocks; ++b)
    {
        auto& block = blocks[b];

        for (auto& z : block.data)
        {
            for (auto& y : z)
            {
                for (auto& x : y)
                {
                    x.distance = clamp(x.distance, -distance, distance);
                }
            }
        }
    }
}

void SparseTSDF::EraseAboveDistance(float threshold)
{
    for (int i = 0; i < current_blocks; ++i)
    {
        auto& b = blocks[i];


        for (int i = 0; i < VOXEL_BLOCK_SIZE; ++i)
        {
            for (int j = 0; j < VOXEL_BLOCK_SIZE; ++j)
            {
                for (int k = 0; k < VOXEL_BLOCK_SIZE; ++k)
                {
                    if (std::abs(b.data[i][j][k].distance) > threshold)
                    {
                        b.data[i][j][k].weight   = 0;
                        b.data[i][j][k].distance = 0;
                    }
                }
            }
        }
    }
}

int SparseTSDF::NumZeroVoxels() const
{
    int n = 0;
    for (int b = 0; b < current_blocks; ++b)
    {
        auto& block = blocks[b];

        for (auto& z : block.data)
        {
            for (auto& y : z)
            {
                for (auto& x : y)
                {
                    if (x.weight == 0) n++;
                }
            }
        }
    }
    return n;
}

int SparseTSDF::NumNonZeroVoxels() const
{
    int n = 0;
    for (int b = 0; b < current_blocks; ++b)
    {
        auto& block = blocks[b];

        for (auto& z : block.data)
        {
            for (auto& y : z)
            {
                for (auto& x : y)
                {
                    if (x.weight != 0) n++;
                }
            }
        }
    }
    return n;
}


void SparseTSDF::SetForAll(float distance, float weight)
{
    for (int b = 0; b < current_blocks; ++b)
    {
        auto& block = blocks[b];

        for (auto& z : block.data)
        {
            for (auto& y : z)
            {
                for (auto& x : y)
                {
                    x.distance = distance;
                    x.weight   = weight;
                }
            }
        }
    }
}


std::ostream& operator<<(std::ostream& strm, const SparseTSDF& tsdf)
{
    size_t mem_blocks = tsdf.blocks.size() * sizeof(SparseTSDF::VoxelBlock);
    size_t mem_hash   = tsdf.first_hashed_block.size() * sizeof(int);

    // Compute some statistics
    std::vector<double> distances;
    std::vector<double> weights;
    for (int b = 0; b < tsdf.current_blocks; ++b)
    {
        auto& block = tsdf.blocks[b];

        for (auto& z : block.data)
            for (auto& y : z)
                for (auto& x : y)
                {
                    if (x.weight > 0)
                    {
                        distances.push_back(x.distance);
                        weights.push_back(x.weight);
                    }
                }
    }

    Statistics d_st(distances);
    Statistics w_st(weights);

    strm << "[SparseTSDF]" << std::endl;
    strm << "  VoxelSize    " << tsdf.voxel_size << std::endl;
    strm << "  hash_size    " << tsdf.hash_size << std::endl;
    strm << "  Blocks       " << tsdf.current_blocks << "/" << tsdf.blocks.size() << std::endl;
    strm << "  Mem Blocks   " << mem_blocks / (1000.0 * 1000) << " MB" << std::endl;
    strm << "  Mem Hash     " << mem_hash / (1000.0 * 1000) << " MB" << std::endl;
    strm << "  Distance     [" << d_st.min << ", " << d_st.max << "]" << std::endl;
    strm << "  Weight       [" << w_st.min << ", " << w_st.max << "]" << std::endl;

    int total     = tsdf.current_blocks * 8 * 8 * 8;
    int nnz       = tsdf.NumNonZeroVoxels();
    float density = nnz / (tsdf.current_blocks * 8.f * 8 * 8);
    strm << "  Block Density  " << nnz << "/" << total << " = " << density * 100 << "%" << std::endl;

    float total_size    = tsdf.Bounds().Volume() * 8 * 8 * 8;
    float total_density = nnz / total_size;
    strm << "  Total Density  " << nnz << "/" << total_size << " = " << total_density * 100 << "%";

    auto bounds = tsdf.Bounds();
    strm << "  Block Bounds " << bounds << std::endl;

    bounds.begin *= 8;
    bounds.end *= 8;
    strm << "  Voxel Bounds " << bounds << std::endl;

    return strm;
}

}  // namespace Saiga
