/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/camera/HDR.h"
#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/exif/TinyEXIF.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/table.h"
#include "saiga/vision/util/Random.h"

#include <fstream>
#include <set>

using namespace Saiga;


TemplatedImage<ucvec3> ExposureImage(ArrayView<double> samples)
{
    TemplatedImage<ucvec3> img(256, 256);
    img.getImageView().set(ucvec3(255, 255, 255));
    SAIGA_ASSERT(samples.size() == 256);

    for (int i = 0; i < 255; ++i)
    {
        vec2 start(i, 255 - samples[i]);
        vec2 end(i + 1, 255 - samples[i + 1]);
        ImageDraw::drawLineBresenham(img.getImageView(), start, end, ucvec3(0, 0, 0));
    }
    return img;
}

float PixelWeight(unsigned char value)
{
    float weight = 0;
    if (value <= 127)
        weight = value;
    else
        weight = 255 - value;
    weight *= (1.f / 256);
    return weight * weight;
}



void DillateOverExposure(ImageView<float> input)
{
    for (int k = 0; k < 2; ++k)
    {
        TemplatedImage<unsigned char> img2(input.dimensions());
        input.copyTo(img2.getImageView());

        for (int i : input.rowRange(1))
        {
            for (int j : input.colRange(1))
            {
                if (input(i, j) == 255)
                {
                    img2(i - 1, j - 1) = 255;
                    img2(i, j - 1)     = 255;
                    img2(i + 1, j - 1) = 255;

                    img2(i - 1, j) = 255;
                    img2(i, j)     = 255;
                    img2(i + 1, j) = 255;

                    img2(i - 1, j + 1) = 255;
                    img2(i, j + 1)     = 255;
                    img2(i + 1, j + 1) = 255;
                }
            }
        }

        img2.getImageView().copyTo(input);
    }
}

class ResponseCalib
{
   public:
    struct ResponseImage
    {
        std::string file;
        double exposure_value;

        TemplatedImage<ucvec3> img;

        // each channel separate + scaled to range [0, 1]
        std::vector<TemplatedImage<float>> separate_channels;
    };
    ResponseCalib(const std::string& dir)
    {
        std::cout << "loading response files from " << dir << std::endl;
        Directory d(dir);
        auto files = d.getFilesEnding(".jpg");
        std::sort(files.begin(), files.end());

        for (auto f : files)
        {
            ResponseImage ri;
            ri.file = dir + "/" + f;


            auto data = File::loadFileBinary(ri.file);
            TinyEXIF::EXIFInfo info;
            info.parseFrom((unsigned char*)data.data(), data.size());
            ri.exposure_value =
                log2((info.FNumber * info.FNumber) / info.ExposureTime) + log2(info.ISOSpeedRatings / 100.0);

            ri.exposure_value = 1. / exp2(ri.exposure_value);
            ri.img.load(ri.file);

            std::cout << std::setw(10) << f << " " << ri.img << " EV: " << ri.exposure_value << std::endl;

            for (int i = 0; i < 3; ++i)
            {
                TemplatedImage<float> chan(ri.img.dimensions());
                for (int y : chan.rowRange())
                {
                    for (int x : chan.colRange())
                    {
                        chan(y, x) = ri.img(y, x)[i];
                    }
                }
                DillateOverExposure(chan);
                ri.separate_channels.push_back(chan);
            }

            images.push_back(ri);
        }
    }

    void CalibrateChannel(int c)
    {
        DiscreteResponseFunction inv_response;
        TemplatedImage<double> irradiance(images.front().img.dimensions());
        TemplatedImage<int> count(irradiance.dimensions());

        {
            irradiance.makeZero();
            count.makeZero();

            // Initialized irradiance with mean
            for (int k = 0; k < images.size(); ++k)
            {
                auto& img = images[k].separate_channels[c];
                for (int i : img.rowRange())
                {
                    for (int j : img.colRange())
                    {
                        irradiance(i, j) += img(i, j);
                        count(i, j)++;
                    }
                }
            }
            for (int i : irradiance.rowRange())
            {
                for (int j : irradiance.colRange())
                {
                    SAIGA_ASSERT(count(i, j) > 0);
                    irradiance(i, j) = irradiance(i, j) / count(i, j);
                }
            }
        }


        for (int i = 0; i < 10; ++i)
        {
            inv_response = EstimateInverseResponse(c, irradiance);
            irradiance   = EstimateIrradiance(c, inv_response);

            // double rescale = 255.0 / inv_response.irradiance[255];
            // for (auto& d : inv_response.irradiance) d *= rescale;
            // for (auto i : irradiance.rowRange())
            //     for (auto j : irradiance.colRange()) irradiance(i, j) *= rescale;


            inv_response.Image().save("exposure_optimized_" + std::to_string(i) + ".png");
        }

        for (int i = 0; i < 256; ++i)
        {
            std::cout << std::setw(4) << i << ": " << inv_response.irradiance[i] << std::endl;
        }

        double irr_mi, irr_ma;
        irradiance.getImageView().findMinMax(irr_mi, irr_ma);
        std::cout << "Min/Max Irradiance: " << irr_mi << " " << irr_ma << std::endl;
    }


    DiscreteResponseFunction<double> EstimateInverseResponse(int c, TemplatedImage<double> irradiance)
    {
        DiscreteResponseFunction result;
        for (auto& d : result.irradiance) d = 0;
        std::vector<int> counts(result.irradiance.size(), 0);

        for (int k = 0; k < images.size(); ++k)
        {
            auto& img = images[k].separate_channels[c];
            for (int i : img.rowRange())
            {
                for (int j : img.colRange())
                {
                    unsigned char color = img(i, j);
                    if (color == 255) continue;

                    result.irradiance[color] += irradiance(i, j) * images[k].exposure_value;
                    counts[color] += 1;
                }
            }
        }

        for (int i = 0; i < result.irradiance.size(); ++i)
        {
            result.irradiance[i] = result.irradiance[i] / counts[i];

            if (!std::isfinite(result.irradiance[i]) && i > 1)
            {
                result.irradiance[i] = result.irradiance[i - 1] + (result.irradiance[i - 1] - result.irradiance[i - 2]);
                std::cout << "filling missing " << i << " " << result.irradiance[i] << std::endl;
            }
        }

#if 0
        DiscreteResponseFunction cpy = result;
        for (int i = 1; i < result.irradiance.size() - 1; ++i)
        {
            // smooth with constraint r'' = 0
            auto v0 = result.irradiance[i - 1];
            auto v1 = result.irradiance[i];
            auto v2 = result.irradiance[i + 1];

            auto w0 = counts[i - 1];
            auto w1 = counts[i ];
            auto w2 = counts[i + 1];

            auto g = (v0 + v2) * 0.5;
            double alpha =  clamp((std::min(w0,w2)) / (w1 + 1e-5), 0 ,1);
            alpha = w1 / (std::min(w0,w2) + w1);
            // alpha = 1;
            v1 = v1 * alpha + g * (1-alpha);
            cpy.irradiance[i] = v1;
        }
        result = cpy;
#endif

        result.normalize(1);

        // extrapolate white point
        // SAIGA_ASSERT(result.irradiance[255] == 0);

        // result.irradiance[255] = result.irradiance[254] + (result.irradiance[254] - result.irradiance[253]);

        //    result.irradiance[255] = result.irradiance[254];
        //    std::cout << "extrapolate " << result.irradiance[253] << " " << result.irradiance[254] << " "
        //              << result.irradiance[255] << std::endl;
        return result;
    }


    TemplatedImage<double> EstimateIrradiance(int c, DiscreteResponseFunction<double> f)
    {
        TemplatedImage<double> result(images.front().img.dimensions());
        TemplatedImage<double> counts(images.front().img.dimensions());
        result.makeZero();

        for (int k = 0; k < images.size(); ++k)
        {
            auto& img = images[k].separate_channels[c];

            for (int i : result.rowRange())
            {
                for (int j : result.colRange())
                {
                    auto color = img(i, j);
                    if (color == 255) continue;


                    float w = PixelWeight(color) * (1. / 255);
                    w = 1;
                    result(i, j) += w * f(color) / images[k].exposure_value;
                    counts(i, j) += w * 1;
                    //result(i, j) += w * f(color);
                    //counts(i, j) += w * images[k].exposure_value;
                }
            }
        }

        for (int i : result.rowRange())
        {
            for (int j : result.colRange())
            {
                result(i, j) = result(i, j) / counts(i, j);
                // SAIGA_ASSERT(std::isfinite(result(i, j)));
                if (result(i, j) < 0) result(i, j) = 0;
            }
        }
        // SAIGA_ASSERT(result.getImageView().isFinite());
        return result;
    }

#if 0
    ResponseCalib()
    {
        std::vector<double> ground_truth;
        {
            std::ifstream strm(dataset_dir + "/pcalib.txt");

            while (!strm.eof())
            {
                std::string str;
                strm >> str;
                double d = Saiga::to_double(str);
                ground_truth.push_back(d);
            }
            std::cout << "Found ground truth: " << ground_truth.size() << std::endl;
            ground_truth.resize(256);
            ExposureImage(ground_truth).save("exposure_gt.png");
        }



        std::vector<double> exposures;
        {
            auto strs = File::loadFileStringArray(dataset_dir + "/times.txt");
            for (auto s : strs)
            {
                auto l = split(s, ' ');
                if (l.empty()) continue;
                SAIGA_ASSERT(l.size() == 3);
                exposures.push_back(to_double(l[2]));
            }
        }

        using ImageType = TemplatedImage<unsigned char>;
        std::vector<ImageType> images;
        {
            auto full_dir = dataset_dir + "/images/";
            Directory dir(full_dir);
            auto image_str = dir.getFilesEnding(".jpg");
            std::sort(image_str.begin(), image_str.end());

            images.resize(image_str.size());

#    pragma omp parallel for num_threads(4)
            for (int i = 0; i < image_str.size(); ++i)
            {
                ImageType img(full_dir + image_str[i]);

                DillateOverExposure(img.getImageView());
#    if 0
                ivec2 block_size(4, 4);
            ImageType small(img.h / block_size.y(), img.w / block_size.x());
            // img.getImageView().copyScaleDownPow2(small.getImageView(), 1);

            for (int i : small.rowRange())
            {
                for (int j : small.colRange())
                {
                    unsigned char v = 0;
                    for (int y = 0; y < block_size(1); ++y)
                    {
                        for (int x = 0; x < block_size(0); ++x)
                        {
                            v = std::max(v, img(i * block_size(1) + y, j * block_size(0) + x));
                        }
                    }
                    small(i, j) = v;
                }
            }
#    endif

                images[i] = img;
            }
            std::cout << "loaded " << images.size() << " images" << std::endl;
        }
        SAIGA_ASSERT(exposures.size() == images.size());
        images[500].save("exposure_test.png");
    }
#endif

    std::vector<ResponseImage> images;
};

int main(int argc, char** argv)
{
    Saiga::EigenHelper::checkEigenCompabitilty<2765>();
    Saiga::Random::setSeed(15235);

    if (argc <= 1)
    {
        std::cout << "Usage and Example: \nsample_vision_calib_response <calib_dir>"
                     "\nsample_vision_calib_response data/calib_narrowGamma_sweep3/"
                  << std::endl;
        return 0;
    }
    std::string dataset_dir = argv[1];

    ResponseCalib rc(dataset_dir);
    rc.CalibrateChannel(0);

#if 0

#endif
    std::cout << "Done." << std::endl;

    return 0;
}
