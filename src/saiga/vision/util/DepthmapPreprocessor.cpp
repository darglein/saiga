/**
 * Copyright (c) 2021 Darius Rückert
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


inline float pixel_footprint(std::size_t x, std::size_t y, float depth, const IntrinsicsPinholed& camera)
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

DMPP::DMPP(const IntrinsicsPinholed& camera, const DMPPParameters& params) : params(params), camera(camera) {}

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



DepthProcessor2::DepthProcessor2(const Settings& settings_in) : settings(settings_in) {}

void DepthProcessor2::remove_occlusion_edges(ImageView<float> depthImageView)
{
    TemplatedImage<vec3> unprojected_image(depthImageView.height, depthImageView.width);
    unprojected_image.makeZero();

    // unproject the depth image
    unproject_depth_image(depthImageView, unprojected_image);

    // create images for the extra data used by the occlusion edge paper (4.2.1)
    TemplatedImage<float> p(depthImageView.height, depthImageView.width);

    // find / delete occlusion edge pixels paper
    compute_image_aspect_ratio(unprojected_image, depthImageView, p);

    // delete pixels from image that surpass the threshold using a looped hysteresis threshold
    use_hysteresis_threshold(depthImageView, unprojected_image, p);
}

void DepthProcessor2::unproject_depth_image(ImageView<const float> depth_imageView, ImageView<vec3> unprojected_image)
{
    int height = depth_imageView.height;
    int width  = depth_imageView.width;

    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            float z                 = depth_imageView(h, w);
            unprojected_image(h, w) = settings.cameraParameters.unproject(vec2(w + 0.5f, h + 0.5f), z);
        }
    }
}


void DepthProcessor2::filter_gaussian(ImageView<const float> input, ImageView<float> output)
{
    if (settings.gauss_radius == 0)
    {
        input.copyTo(output);
    }

    std::vector<float> filter((settings.gauss_radius * 2) + 1);
    for (int i = -settings.gauss_radius; i <= settings.gauss_radius; ++i)
    {
        filter[i + settings.gauss_radius] =
            1.0f / (sqrt(2.0f * pi<float>()) * settings.gauss_standard_deviation) *
            std::exp(-(i * i / (2.0f * settings.gauss_standard_deviation * settings.gauss_standard_deviation)));
    }
    int filter_mid = filter.size() / 2;

    // a temporal image for the first filter pass
    TemplatedImage<float> temp_image(input.height, input.width);

    // go through the pixels and filter in x-direction
    for (int h = 0; h < input.height; ++h)
    {
        for (int w = 0; w < input.width; ++w)
        {
            // if the current pixel already is broken I don't want a new value for it
            if (input(h, w) == settings.broken_values) continue;

            float weights = 0.0f;
            float value   = 0.0f;

            // filter for the current pixel
            float cur_weight = filter[filter_mid];
            weights += cur_weight;
            value += cur_weight * input(h, w);

            // filter to the left until radius is reached or a broken vertex is found
            for (int x = -1; x >= -settings.gauss_radius; --x)
            {
                if ((w + x) < 0 || input(h, w + x) == settings.broken_values) break;

                cur_weight = filter[filter_mid + x];
                weights += cur_weight;
                value += cur_weight * input(h, w + x);
            }

            // filter to the right until radius is reached or a broken vertex is found
            for (int x = 1; x <= settings.gauss_radius; ++x)
            {
                if ((w + x) >= input.width || input(h, w + x) == settings.broken_values) break;

                cur_weight = filter[filter_mid + x];
                weights += cur_weight;
                value += cur_weight * input(h, w + x);
            }

            temp_image(h, w) = value / weights;
        }
    }

    // go through the pixels and filter in y-direction
    for (int h = 0; h < input.height; ++h)
    {
        for (int w = 0; w < input.width; ++w)
        {
            // if the current pixel already is broken I don't want a new value for it
            if (temp_image(h, w) == settings.broken_values) continue;

            float weights = 0.0f;
            float value   = 0.0f;

            // filter for the current pixel
            float cur_weight = filter[filter_mid];
            weights += cur_weight;
            value += cur_weight * input(h, w);

            // filter to the left until radius is reached or a broken vertex is found
            for (int y = -1; y >= -settings.gauss_radius; --y)
            {
                if ((h + y) < 0 || input(h + y, w) == settings.broken_values) break;

                cur_weight = filter[filter_mid + y];
                weights += cur_weight;
                value += cur_weight * input(h + y, w);
            }

            // filter to the right until radius is reached or a broken vertex is found
            for (int y = 1; y <= settings.gauss_radius; ++y)
            {
                if ((h + y) >= input.height || input(h + y, w) == settings.broken_values) break;

                cur_weight = filter[filter_mid + y];
                weights += cur_weight;
                value += cur_weight * input(h + y, w);
            }

            output(h, w) = value / weights;
        }
    }
}

// --- PRIVATE ---

float DepthProcessor2::compute_quad_max_aspect_ratio(const vec3& left_up, const vec3& right_up, const vec3& left_down,
                                                     const vec3& right_down)
{
    // all edge lengths
    float len_up, len_right, len_down, len_left, len_diag_0, len_diag_1;
    len_up    = (left_up - right_up).stableNorm();
    len_right = (right_up - right_down).stableNorm();
    len_down  = (right_down - left_down).stableNorm();
    len_left  = (left_down - left_up).stableNorm();

    len_diag_0 = (left_up - right_down).stableNorm();
    len_diag_1 = (right_up - left_down).stableNorm();

    // edge direction: left up to right down
    float aspect_0 =
        std::max(std::max(len_left, len_down), len_diag_0) / std::min(std::min(len_left, len_down), len_diag_0);
    float aspect_1 =
        std::max(std::max(len_right, len_up), len_diag_0) / std::min(std::min(len_right, len_up), len_diag_0);
    // edge direction: left down to right up
    float aspect_2 =
        std::max(std::max(len_left, len_up), len_diag_1) / std::min(std::min(len_left, len_up), len_diag_1);
    float aspect_3 =
        std::max(std::max(len_right, len_down), len_diag_1) / std::min(std::min(len_right, len_down), len_diag_1);

    // edge direction: left up to right down
    float max_aspect_0 = std::max(aspect_0, aspect_1);
    // edge direction: left down to right up
    float max_aspect_1 = std::max(aspect_2, aspect_3);

    // choose the smaller maximum
    return std::min(max_aspect_0, max_aspect_1);
}

void DepthProcessor2::compute_image_aspect_ratio(ImageView<const vec3> image, ImageView<float> depthImageView,
                                                 ImageView<float> p)
{
    // get disparity data
    TemplatedImage<float> disparity(depthImageView.h, depthImageView.w);
    float median_disparity = get_median_disparity(depthImageView, disparity);

    int height       = disparity.height;
    int width        = disparity.width;
    int quads_height = height - 1;
    int quads_width  = width - 1;

    // temporay information on the p per quad (maximum of aspect ratio using the better triangulation)
    TemplatedImage<float> quad_p(height - 1, width - 1);

    // check quad properties: quad_p
    for (int h = 0; h < quads_height; ++h)
    {
        for (int w = 0; w < quads_width; ++w)
        {
            // check if any vertex has broken depth
            if (image(h, w)[2] == settings.broken_values || image(h + 1, w)[2] == settings.broken_values ||
                image(h, w + 1)[2] == settings.broken_values || image(h + 1, w + 1)[2] == settings.broken_values)
            {
                // the quad has broken depth
                quad_p(h, w) = settings.broken_values;
                continue;
            }

            // if not then get the worse aspect ratio using the better triangulation
            float q_p =
                compute_quad_max_aspect_ratio(image(h, w), image(h, w + 1), image(h + 1, w), image(h + 1, w + 1));
            quad_p(h, w) = q_p;

            // compute the data for the next pixel
            if (h == 0 || w == 0)
            {
                // not needed as hysteresis will ignore edges
            }
            else
            {
                // find the maximum p for this pixel (highest aspect ratio --> highest error)

                // initialize p at current pixel
                float quad_left_up    = quad_p(h - 1, w - 1);
                float quad_left_down  = quad_p(h, w - 1);
                float quad_right_up   = quad_p(h - 1, w);
                float quad_right_down = quad_p(h, w);

                // check the surroundings for p (quads) and d_p (vertices)

                // if any of those contains a broken value propagate it, ...
                if (quad_left_up == settings.broken_values || quad_left_down == settings.broken_values ||
                    quad_right_up == settings.broken_values || quad_right_down == settings.broken_values)
                {
                    p(h, w) = settings.broken_values;
                    continue;
                }
                // else choose the max
                p(h, w) = std::max(std::max(std::max(quad_left_up, quad_left_down), quad_right_up), quad_right_down);

                // --- find d_P and d_D per pixel ---

                // initialize the surrounding pixels for calculating the normal
                vec3 left  = image(h, w - 1);
                vec3 right = image(h, w + 1);
                vec3 up    = image(h - 1, w);
                vec3 down  = image(h + 1, w);

                vec3 viewing = -image(h, w);
                viewing.normalize();
                vec3 normal = cross(right - left, up - down);
                normal.normalize();

                // --- find d_D per pixel ---
                float pixel_d_D = std::min(2.0f, std::max(0.5f, disparity(h, w) / median_disparity));

                // --- p ---
                p(h, w) *= pixel_d_D;
            }
        }
    }
}

float DepthProcessor2::get_median_disparity(ImageView<float> depth_imageView, ImageView<float> disparity_imageView)
{
    // median disparity (broken pixels will not be used)
    std::vector<float> disparities;
    disparities.reserve(disparity_imageView.height * disparity_imageView.width);

    for (int h = 0; h < depth_imageView.height; ++h)
    {
        for (int w = 0; w < depth_imageView.width; ++w)
        {
            float depth = depth_imageView(h, w);

            // check for broken depth
            if (depth == settings.broken_values)
            {
                disparity_imageView(h, w) = 0;
                continue;
            }

            disparity_imageView(h, w) = settings.cameraParameters.bf / depth;
            disparities.push_back(disparity_imageView(h, w));
        }
    }

    std::sort(disparities.begin(), disparities.end());
    int disp_len = disparities.size();
    if (disp_len > 0)
    {
        if (disp_len % 2 == 0)
        {
            return (disparities[(disp_len / 2) - 1] + disparities[disp_len / 2]) / 2;
        }
        else
        {
            return disparities[disp_len / 2];
        }
    }
    return 0.0f;
}

void DepthProcessor2::use_hysteresis_threshold(ImageView<float> depth_image, ImageView<vec3> unprojected_image,
                                               ImageView<float> computed_values)
{
    int height = depth_image.height;
    int width  = depth_image.width;
    std::vector<bool> sure_edges_vec(height * width, false);

    // find the sure-edges
    for (int h = 1; h < height - 1; ++h)
    {
        for (int w = 1; w < width - 1; ++w)
        {
            // everything above maxVal or broken is sure to be part of the edge
            if (computed_values(h, w) >= settings.hyst_max || computed_values(h, w) == settings.broken_values)
            {
                depth_image(h, w)             = settings.broken_values;
                unprojected_image(h, w)       = vec3::Zero();
                sure_edges_vec[h * width + w] = true;
            }
        }
    }

    // loop the whole process until nothing changes anymore
    bool something_changed = true;
    while (something_changed)
    {
        something_changed = false;

        // check for all unsure edges if a sure edge is among its neighbours
        for (int h = 1; h < depth_image.height - 1; ++h)
        {
            for (int w = 1; w < depth_image.width - 1; ++w)
            {
                // everything between minVal and maxVal may be part of the edge
                if (computed_values(h, w) > settings.hyst_min && computed_values(h, w) < settings.hyst_max &&
                    sure_edges_vec[h * width + w] == false)
                {
                    // check the 8 neighbours for sure-edges
                    bool sure_edge_neighbour =
                        sure_edges_vec[(h - 1) * width + w - 1] || sure_edges_vec[(h - 1) * width + w] ||
                        sure_edges_vec[(h - 1) * width + w + 1] || sure_edges_vec[h * width + w - 1] ||
                        sure_edges_vec[h * width + w + 1] || sure_edges_vec[(h + 1) * width + w - 1] ||
                        sure_edges_vec[(h + 1) * width + w] || sure_edges_vec[(h + 1) * width + w + 1];

                    if (sure_edge_neighbour)
                    {
                        // there is a neighbour that is a sure edge --> this is an edge too
                        depth_image(h, w)             = settings.broken_values;
                        unprojected_image(h, w)       = vec3::Zero();
                        sure_edges_vec[h * width + w] = true;
                        something_changed             = true;
                    }
                }
            }
        }
    }
}

void DepthProcessor2::Settings::imgui()
{
    ImGui::InputFloat("hyst min", &hyst_min);
    ImGui::InputFloat("hyst max", &hyst_max);
    ImGui::InputFloat("gauss standard deviation", &gauss_standard_deviation);
    ImGui::InputInt("gauss radius", &gauss_radius);
}

}  // namespace Saiga
