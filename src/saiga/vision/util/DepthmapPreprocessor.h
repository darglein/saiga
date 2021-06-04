/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/image.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
struct SAIGA_VISION_API DMPPParameters
{
    bool apply_downscale = false;
    int downscaleFactor  = 2;  // 2,4,8,16,...

    bool apply_filter    = false;
    int filterRadius     = 3;
    float sigmaFactor    = 50.0f;
    int filterIterations = 1;

    bool apply_holeFilling = false;
    int holeFillIterations = 5;
    float fillDDscale      = 0.5f;

    float dd_factor = 10.0f;



    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


    void renderGui();
};



class SAIGA_VISION_API DMPP
{
   public:
    using DepthMap = ImageView<float>;


    DMPPParameters params;
    DMPP(const IntrinsicsPinholed& camera = IntrinsicsPinholed(), const DMPPParameters& params = DMPPParameters());


    void operator()(DepthMap src, DepthMap dst);
    // Inplace preprocessing
    void operator()(DepthMap src);

    void setCamera(const IntrinsicsPinholed& c) { camera = c; }

    static void scaleDown2median(DepthMap src, DepthMap dst);
    void fillHoles(DepthMap vsrc, DepthMap vdst);
    void applyFilterToImage(DepthMap vsrc, DepthMap vdst);
    void computeMinMax(DepthMap vsrc, float& dmin, float& dmax);

    void renderGui();

   private:
    IntrinsicsPinholed camera;
};


class SAIGA_VISION_API DepthProcessor2
{
   public:
    struct Settings
    {
        // the value used for pixels that contain failed depth measurements or got discarded
        float broken_values = 0.0f;

        // options for the gaussian filtering
        int gauss_radius               = 4;
        float gauss_standard_deviation = 1.2f;

        // options for using the hysteresis threshold
        // if a pixel in the filtered image...
        // ... is greater than hyst_max, it gets deleted
        // ... is lower than hyst_min, it doesn't get deleted
        // ... lies between those values, it gets deleted only if a neighbouring pixel gets deleted. This may also
        // propagate across the ismage
        float hyst_min = 12.0f;
        float hyst_max = 23.0f;

        // camera parameters of the camera that took the picture which gets processed
        StereoCamera4Base<float> cameraParameters;

        void imgui();
    };

    DepthProcessor2(const Settings& settings_in);

    void Process(ImageView<float> depthImageView)
    {
        remove_occlusion_edges(depthImageView);
        filter_gaussian(depthImageView, depthImageView);
    }

    // combines most of the
    void remove_occlusion_edges(ImageView<float> depthImageView);

    // unprojects a depth image
    // unprojected_image must be at least as big as depth_image
    void unproject_depth_image(ImageView<const float> depth_imageView, ImageView<vec3> unprojected_image);


    // works in-place!
    void filter_gaussian(ImageView<const float> input, ImageView<float> output);

   private:
    Settings settings;

    // computes the aspect ratios for all possible triangulations of a quad and returns the better triangulation and the
    // worse aspect ratio from those triangles
    // -- input --
    // quad:	left_up		right_up
    //			left_down	right_down
    // -- output --
    // the aspect ratio
    //
    // aspect ratio = max(a, b, c) / min(a, b, c)
    // --> this ratio works best for occlusion edge detection (also mentioned in the paper under 4.2.1)
    float compute_quad_max_aspect_ratio(const vec3& left_up, const vec3& right_up, const vec3& left_down,
                                        const vec3& right_down);

    // computes aspect ratio information for a depth image for later triangulation and vertex deletion:
    // p per vertex:
    //		The minimum of the 4 corresponding quads of each vertex.
    //		There are 2 aspect ratios (one per triangle) per such quad.
    //		Each quad uses the triangulation that minimizes the maximum aspect ratio.
    //
    // the edge of the image is ignored
    void compute_image_aspect_ratio(ImageView<const vec3> image, ImageView<float> depthImageView, ImageView<float> p);

    // gets a depth image and fills a disparity image. Returns the average disparity
    float get_median_disparity(ImageView<float> depth_image, ImageView<float> disparity_image);

    // deletes pixels according to the hysteresis threshold https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    // adds unclear edge Pixels with sure edge neighbours to sure edges,
    // loops until nothing changes anymore and ignores the edge of the image
    //
    // additionally removes the vertices from unprojected_image whose pixels get deleted from depth_image
    //
    // values above maxVal are sure to be edges
    // values below minVal are sure to be non-edges WITH EXCEPTION OF brokenVal which is a sure edge
    void use_hysteresis_threshold(ImageView<float> depth_image, ImageView<vec3> unprojected_image,
                                  ImageView<float> computed_values);
};

}  // namespace Saiga
