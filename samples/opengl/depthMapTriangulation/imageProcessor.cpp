/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "imageProcessor.h"

namespace Saiga
{

// --- PUBLIC ---

ImageProcessor::ImageProcessor(const Settings& settings_in) : settings(settings_in) {}

void ImageProcessor::remove_occlusion_edges(ImageView<float> depthImageView)
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

void ImageProcessor::unproject_depth_image(ImageView<const float> depth_imageView,
                                                  ImageView<vec3> unprojected_image)
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

void ImageProcessor::unproject_depth_image(ImageView<const float> depth_imageView,
                                                  ImageView<OpenMesh::Vec3f> unprojected_image)
{
    int height = depth_imageView.height;
    int width  = depth_imageView.width;

    float z;
    for (int h = 0; h < height; ++h)
    {
        for (int w = 0; w < width; ++w)
        {
            z                       = depth_imageView(h, w);
            vec3 unprojected_point  = settings.cameraParameters.unproject(vec2(w + 0.5f, h + 0.5f), z);
            unprojected_image(h, w) = OpenMesh::Vec3f(unprojected_point[0], unprojected_point[1], unprojected_point[2]);
        }
    }
}

void ImageProcessor::filter_gaussian(ImageView<const float> input, ImageView<float> output)
{
    std::vector<float> filter((settings.gauss_radius * 2) + 1);
    for (int i = -settings.gauss_radius; i <= settings.gauss_radius; ++i)
    {
        filter[i + settings.gauss_radius] = 1.0f / (sqrt(2.0f * pi<float>()) * settings.gauss_stadard_deviation) *
                                            std::exp(-(i * i / (2.0f * settings.gauss_stadard_deviation * settings.gauss_stadard_deviation)));
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

float ImageProcessor::compute_quad_max_aspect_ratio(const vec3& left_up, const vec3& right_up, const vec3& left_down, const vec3& right_down)
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

    // choose the smaller maximum and its triangulation
    if (max_aspect_0 <= max_aspect_1)
    {
        return max_aspect_0;
    }
    else
    {
        return max_aspect_1;
    }
}

void ImageProcessor::compute_image_aspect_ratio(ImageView<const vec3> image,
                                                       ImageView<float> depthImageView, ImageView<float> p)
{
    // get disparity data
    TemplatedImage<float> disparity(depthImageView.h, depthImageView.w);
    float median_disparity     = get_median_disparity(depthImageView, disparity);

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

float ImageProcessor::get_median_disparity(ImageView<float> depth_imageView, ImageView<float> disparity_imageView)
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

void ImageProcessor::use_hysteresis_threshold(ImageView<float> depth_image, ImageView<vec3> unprojected_image,
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

} // namespace saiga
