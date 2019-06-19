/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "DepthmapPreprocessor.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/ini/ini.h"


namespace Saiga
{
inline bool dm_is_depthdisc(float* widths, float* depths, float dd_factor, int i1, int i2, float dis)
{
    /* Find index that corresponds to smaller depth. */
    int i_min = i1;
    int i_max = i2;
    if (depths[i2] < depths[i1]) std::swap(i_min, i_max);

    /* Check if indices are a diagonal. */

    dd_factor *= dis;

    /* Check for depth discontinuity. */
    if (depths[i_max] - depths[i_min] > widths[i_min] * dd_factor) return true;

    return false;
}


inline float pixel_footprint(std::size_t x, std::size_t y, float depth, const Intrinsics4& camera)
{
#ifdef SPHERICAL_DEPTH
    vec3 ray = invproj * vec3((float)x + 0.5f, (float)y + 0.5f, 1.0f);
    return invproj[0][0] * depth / length(ray);
#else
    //    vec3 ray = invproj * vec3
    //            ((float)x + 0.5f, (float)y + 0.5f, 1.0f);
    //    return invproj[0][0] * depth;
    return 1.0 / camera.fx * depth;
#endif
}

void DMPPParameters::fromConfigFile(const std::string& file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());



    apply_downscale   = ini.GetAddBool("dmpp", "apply_downscale", apply_downscale);
    apply_filter      = ini.GetAddBool("dmpp", "apply_filter", apply_filter);
    apply_holeFilling = ini.GetAddBool("dmpp", "apply_holeFilling", apply_holeFilling);

    downscaleFactor    = ini.GetAddLong("dmpp", "downscaleFactor", downscaleFactor);
    filterRadius       = ini.GetAddLong("dmpp", "filterRadius", filterRadius);
    filterIterations   = ini.GetAddLong("dmpp", "filterIterations", filterIterations);
    holeFillIterations = ini.GetAddLong("dmpp", "holeFillIterations", holeFillIterations);

    sigmaFactor = ini.GetAddDouble("dmpp", "sigmaFactor", sigmaFactor);
    fillDDscale = ini.GetAddDouble("dmpp", "fillDDscale", fillDDscale);
    dd_factor   = ini.GetAddDouble("dmpp", "dd_factor", dd_factor);


    if (ini.changed()) ini.SaveFile(file.c_str());
}

void DMPPParameters::renderGui()
{
    ImGui::Checkbox("apply_downscale", &apply_downscale);
    if (apply_downscale)
    {
        ImGui::InputInt("downscaleFactor", &downscaleFactor);
    }

    ImGui::Checkbox("apply_filter", &apply_filter);
    if (apply_filter)
    {
        ImGui::SliderInt("filterRadius", &filterRadius, 0, 5);
        ImGui::InputFloat("sigmaFactor", &sigmaFactor);
        ImGui::SliderInt("filterIterations", &filterIterations, 0, 5);
    }


    ImGui::Checkbox("apply_holeFilling", &apply_holeFilling);
    if (apply_holeFilling)
    {
        ImGui::SliderInt("holeFillIterations", &holeFillIterations, 0, 50);
        ImGui::SliderFloat("fillDDscale", &fillDDscale, 0, 5);
    }

    ImGui::InputFloat("dd_factor", &dd_factor);
}

DMPP::DMPP(const Intrinsics4& camera, const DMPPParameters& params) : params(params), camera(camera) {}

void DMPP::operator()(DepthMap _src, DepthMap dst)
{
    static thread_local TemplatedImage<float> tmp;
    tmp.create(dst.h, dst.w);

    auto src = _src;

    if (params.apply_downscale && _src.w / 2 == dst.w)
    {
        scaleDown2median(src, tmp.getImageView());
        src = tmp.getImageView();
    }

    if (params.apply_filter)
    {
        applyFilterToImage(src, dst);
    }
    else
    {
        src.copyTo(dst);
    }

    if (params.apply_holeFilling)
    {
        fillHoles(dst, dst);
    }
}


void DMPP::operator()(DepthMap src)
{
    SAIGA_ASSERT(!params.apply_downscale, "Inplace preprocessing cannot be applied with downscale!");

    static thread_local TemplatedImage<float> tmp;
    tmp.create(src.h, src.w);


    if (params.apply_filter)
    {
        applyFilterToImage(src, tmp);
        tmp.getImageView().copyTo(src);
    }

    if (params.apply_holeFilling)
    {
        fillHoles(src, src);
    }
}

template <typename T>
std::vector<T> gaussianBlurKernel2D(int radius, T sigmaX, T sigmaY)
{
    const int ELEMENTS = (radius * 2 + 1) * (radius * 2 + 1);
    std::vector<T> kernel(ELEMENTS);

    T ivar2X = 1.0f / (2.0f * sigmaX * sigmaX);
    T ivar2Y = 1.0f / (2.0f * sigmaY * sigmaY);

    T kernelSum(0);

    for (int i = -radius; i <= radius; i++)
    {
        for (int j = -radius; j <= radius; j++)
        {
            int idx     = (j + radius) + (i + radius) * (radius * 2 + 1);
            kernel[idx] = std::exp(-j * j * ivar2X) + std::exp(-i * i * ivar2Y);
            kernelSum += kernel[idx];
        }
    }
    // normalize
    for (T& k : kernel)
    {
        k /= kernelSum;
    }
    return kernel;
}

void DMPP::applyFilterToImage(DepthMap vsrc, DepthMap vdst)
{
#ifdef FF_PRINT_TIMINGS
    ScopedTimerPrint tim("applyFilterToImage");
#endif

    SAIGA_ASSERT(vsrc.width == vdst.width && vsrc.height == vdst.height);
    //    SAIGA_ASSERT(filterRadius == 1);



    std::vector<float> kernel = gaussianBlurKernel2D(params.filterRadius, params.sigmaFactor, params.sigmaFactor);
    ImageView<float> kernelI((params.filterRadius * 2 + 1), (params.filterRadius * 2 + 1), kernel.data());

    for (int it = 0; it < params.filterIterations; ++it)
    {
        // filter
        //#pragma omp parallel for
        for (int i = 0; i < vdst.height; ++i)
        {
            for (int j = 0; j < vdst.width; ++j)
            {
                float d = vsrc(i, j);
#if 1
                float widths[2];
                float depths[2];
                depths[0] = d;
                widths[0] = pixel_footprint(j, i, d, camera);
#endif

                int size = params.filterRadius;

                float w = kernelI(params.filterRadius, params.filterRadius);
                //            w = 0;
                float wsum = w;
                float zsum = d * w;

                //                int connect = con(i,j);

                int k = 0;
                for (int di = -size; di <= size; ++di)
                {
                    for (int dj = -size; dj <= size; ++dj)
                    {
                        if (di == 0 && dj == 0) continue;

                        float d2 = vsrc.clampedRead(i + di, j + dj);
#if 1
                        vec2 offset(di, dj);
                        float du2 = dot(offset, offset);
                        //                        float dz = std::abs(d2 - d);
                        //                        float dz = abs(d2 - d);
                        depths[1] = d2;
                        widths[1] = pixel_footprint(j + dj, i + di, d2, camera);


                        if (d2 > 0 && !dm_is_depthdisc(widths, depths, params.dd_factor, 0, 1, sqrt(du2)))
                        {
                            w = kernelI(di + params.filterRadius, dj + params.filterRadius);
                        }
                        else
                        {
                            w = 0;
                        }
#else
                        if (connect & (1 << k))
                        {
                            w = kernelI(di + filterRadius, dj + filterRadius);
                        }
                        else
                        {
                            w = 0;
                        }
#endif
                        wsum += w;
                        zsum += w * d2;
                        k++;
                    }
                }
                d          = zsum / wsum;
                vdst(i, j) = d;
            }
        }
    }
}

void DMPP::computeMinMax(DepthMap vsrc, float& dmin, float& dmax)
{
    dmin = 5345345;
    dmax = -345345435;
    // look for min/max
    for (int i = 0; i < vsrc.height; ++i)
    {
        for (int j = 0; j < vsrc.width; ++j)
        {
            float d = vsrc(i, j);
            if (d == 0)
            {
                d = 0;
            }
            else
            {
                dmin = std::min(dmin, d);
                dmax = std::max(dmax, d);
            }
        }
    }
    dmin -= 0.001f;
    dmax += 0.001f;
}



void DMPP::fillHoles(DepthMap vsrc, DepthMap vdst)
{
    //    std::cout << "fill holes " << params.holeFillIterations << std::endl;
    SAIGA_ASSERT(vsrc.width == vdst.width && vsrc.height == vdst.height);


    std::vector<char> mask(vsrc.width * vsrc.height);
    ImageView<char> vmask(vsrc.height, vsrc.width, mask.data());

    for (int i = 0; i < vsrc.height; ++i)
    {
        for (int j = 0; j < vsrc.width; ++j)
        {
            vmask(i, j) = 0;
            if (vsrc(i, j) == 0) vmask(i, j) = 1;
        }
    }



    for (int it = 0; it < params.holeFillIterations; ++it)
    {
        for (int i = 0; i < vsrc.height; ++i)
        {
            for (int j = 0; j < vsrc.width; ++j)
            {
                if (vmask(i, j))
                {
                    float du = vsrc.clampedRead(i + 1, j);
                    float db = vsrc.clampedRead(i - 1, j);
                    float dl = vsrc.clampedRead(i, j + 1);
                    float dr = vsrc.clampedRead(i, j - 1);

                    float sum = 0;
                    float w   = 0;

                    if (du > 0)
                    {
                        sum += du;
                        w += 1;
                    }
                    if (db > 0)
                    {
                        sum += db;
                        w += 1;
                    }
                    if (dl > 0)
                    {
                        sum += dl;
                        w += 1;
                    }
                    if (dr > 0)
                    {
                        sum += dr;
                        w += 1;
                    }

                    if (w > 0) vsrc(i, j) = sum / w;
                }
            }
        }
    }


    for (int i = 0; i < vsrc.height; ++i)
    {
        for (int j = 0; j < vsrc.width; ++j)
        {
            if (vmask(i, j))
            {
                // check if we actually filled a hole instead of just extruding and edge
                int found = 0;
                for (int x = -params.holeFillIterations; x < 0; ++x)
                    if (vmask.clampedRead(i, j + x) == 0)
                    {
                        found++;
                        break;
                    }

                for (int x = 1; x <= params.holeFillIterations; ++x)
                    if (vmask.clampedRead(i, j + x) == 0)
                    {
                        found++;
                        break;
                    }

                for (int y = -params.holeFillIterations; y < 0; ++y)
                    if (vmask.clampedRead(i + y, j) == 0)
                    {
                        found++;
                        break;
                    }

                for (int y = 1; y <= params.holeFillIterations; ++y)
                    if (vmask.clampedRead(i + y, j) == 0)
                    {
                        found++;
                        break;
                    }

                if (found < 3)
                {
                    vsrc(i, j) = 0;
                    continue;
                }

                // check for depth discontinuity with stronger dd factor

                float widths[5];
                float depths[5];
                depths[0] = vsrc(i, j);
                widths[0] = pixel_footprint(j, i, depths[0], camera);

                depths[1] = vsrc.clampedRead(i + 1, j);
                depths[2] = vsrc.clampedRead(i - 1, j);
                depths[3] = vsrc.clampedRead(i, j + 1);
                depths[4] = vsrc.clampedRead(i, j - 1);


                widths[1] = pixel_footprint(j, i + 1, depths[1], camera);
                widths[2] = pixel_footprint(j, i - 1, depths[2], camera);
                widths[3] = pixel_footprint(j + 1, i, depths[3], camera);
                widths[4] = pixel_footprint(j - 1, i, depths[4], camera);

                for (int k = 0; k < 4; ++k)
                {
                    if (dm_is_depthdisc(widths, depths, params.dd_factor * params.fillDDscale, 0, k + 1, 1))
                    {
                        vsrc(i, j) = 0;
                        break;
                    }
                }
            }
        }
    }
}

void DMPP::scaleDown2median(DepthMap src, DepthMap dst)
{
    SAIGA_ASSERT(src.width == 2 * dst.width && src.height == 2 * dst.height);


    for (int i = 0; i < dst.height; ++i)
    {
        for (int j = 0; j < dst.width; ++j)
        {
            std::array<float, 4> vs;
            for (int di = 0; di < 2; ++di)
            {
                for (int dj = 0; dj < 2; ++dj)
                {
                    vs[di * 2 + dj] = src(i * 2 + di, j * 2 + dj);
                }
            }
            std::sort(vs.begin(), vs.end());
            dst(i, j) = vs[1];
        }
    }
}



void DMPP::renderGui()
{
    if (ImGui::CollapsingHeader("DMPreprocessor"))
    {
        params.renderGui();
        ImGui::Separator();
    }
}


}  // namespace Saiga
