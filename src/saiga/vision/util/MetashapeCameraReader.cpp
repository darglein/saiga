/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "MetashapeCameraReader.h"

#include "saiga/core/util/file.h"
#include "saiga/core/util/tinyxml2.h"
#include "saiga/core/util/tostring.h"

using namespace tinyxml2;

void print_tree(const tinyxml2::XMLElement* element, int depth = 0)
{
    if (!element) return;
    for (int d = 0; d < depth; ++d) std::cout << " ";
    std::cout << element->Name() << std::endl;
    //    print_tree(element->FirstChildElement(), depth++);

    for (const XMLNode* node = element->FirstChild(); node; node = node->NextSibling())
    {
        auto el = node->ToElement();
        print_tree(el, depth + 1);
    }
}
namespace Saiga
{
MetashapeCameraReader::MetashapeCameraReader(const std::string& file, bool verbose)
{
    std::cout << "Reading metashape file " << file << std::endl;

    tinyxml2::XMLDocument doc;
    doc.LoadFile(file.c_str());


    auto chunk_root = doc.RootElement()->FirstChildElement("chunk");
    SAIGA_ASSERT(chunk_root);

    // only one chunk exists
    SAIGA_ASSERT(chunk_root == doc.RootElement()->LastChildElement("chunk"));

    {
        // intrinsics
        auto sensors_node = chunk_root->FirstChildElement("sensors");

        for (const XMLNode* node = sensors_node->FirstChild(); node; node = node->NextSibling())
        {
            auto el = node->ToElement();
            if (std::string(el->Name()) == "sensor")
            {
                Intrinsics si;
                auto calib = el->FirstChildElement("calibration");

                si.w = calib->FirstChildElement("resolution")->IntAttribute("width");
                si.h = calib->FirstChildElement("resolution")->IntAttribute("height");

                si.K.cx = calib->FirstChildElement("cx")->DoubleText();
                si.K.cy = calib->FirstChildElement("cy")->DoubleText();

                si.K.cx += si.w * 0.5;
                si.K.cy += si.h * 0.5;


                double f  = calib->FirstChildElement("f")->DoubleText();
                double b1 = calib->FirstChildElement("b1")->DoubleText();
                double b2 = calib->FirstChildElement("b2")->DoubleText();

                si.K.fx = f + b1;
                si.K.s  = b2;
                si.K.fy = f;

                si.dis.k1 = calib->FirstChildElement("k1")->DoubleText();
                si.dis.k2 = calib->FirstChildElement("k2")->DoubleText();
                si.dis.k3 = calib->FirstChildElement("k3")->DoubleText();
                si.dis.p1 = calib->FirstChildElement("p1")->DoubleText();
                si.dis.p2 = calib->FirstChildElement("p2")->DoubleText();

                si.name = el->Attribute("label");

                if (verbose)
                {
                    std::cout << "Found sensor " << si.name << std::endl;
                    std::cout << si.w << "x" << si.h << std::endl;
                    std::cout << si.K << std::endl;
                    std::cout << si.dis.Coeffs().transpose() << std::endl;
                }
                sensors.push_back(si);
            }
        }
    }

    {
        // extrinsics
        auto sensors_node = chunk_root->FirstChildElement("cameras");

        for (const XMLNode* node = sensors_node->FirstChild(); node; node = node->NextSibling())
        {
            auto el = node->ToElement();
            if (std::string(el->Name()) == "camera")
            {
                Extrinsics ex;
                ex.name      = el->Attribute("label");
                ex.sensor_id = el->IntAttribute("sensor_id");

                {
                    auto orien = el->FirstChildElement("orientation");
                    if (orien)
                    {
                        ex.orientation = orien->IntText();
                    }
                }

                auto t_str = el->FirstChildElement("transform")->GetText();
                auto t_els = Saiga::split(t_str, ' ');
                SAIGA_ASSERT(t_els.size() == 16);

                Mat4 T;
                for (int i = 0; i < 16; ++i)
                {
                    double d        = to_double(t_els[i]);
                    T(i / 4, i % 4) = d;
                }

                ex.pose = SE3::fitToSE3(T);
                cameras.push_back(ex);

                if (verbose)
                {
                    std::cout << "Found camera " << ex.name;
                    std::cout << "  sensor " << ex.sensor_id;
                    std::cout << "  orientation " << ex.orientation;
                    std::cout << "  pose " << ex.pose << std::endl;
                }
            }
        }
    }

    {
        // transform node
        auto transform_node = chunk_root->FirstChildElement("transform");
        if (transform_node)
        {
            auto r_str = transform_node->FirstChildElement("rotation")->GetText();
            auto r_els = Saiga::split(r_str, ' ');
            SAIGA_ASSERT(r_els.size() == 9);

            auto t_str = transform_node->FirstChildElement("translation")->GetText();
            auto t_els = Saiga::split(t_str, ' ');
            SAIGA_ASSERT(t_els.size() == 3);

            auto s_str = transform_node->FirstChildElement("scale")->GetText();
            auto s_els = Saiga::split(s_str, ' ');
            SAIGA_ASSERT(s_els.size() == 1);

            Mat3 R;
            for (int i = 0; i < 9; ++i)
            {
                double d        = to_double(r_els[i]);
                R(i / 3, i % 3) = d;
            }

            Vec3 t;
            for (int i = 0; i < 3; ++i)
            {
                double d = to_double(t_els[i]);
                t(i)     = d;
            }

            double s = to_double(s_els.front());

            auto T    = SE3(R, t);
            auto Tsim = DSim3(T, s);

            for (auto& c : cameras)
            {
                auto ct = DSim3(c.pose, 1);
                c.pose  = (Tsim * ct).se3();
            }

            if (verbose)
            {
                std::cout << "Transformation" << std::endl;
                std::cout << R << std::endl;
                std::cout << t.transpose() << std::endl;
                std::cout << s << std::endl;
            }
        }
    }

    // print_tree(chunk_root);
}
}  // namespace Saiga