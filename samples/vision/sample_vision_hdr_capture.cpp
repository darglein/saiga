/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/framework/framework.h"
#include "saiga/core/geometry/half_edge_mesh.h"
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/file.h"

#include <fstream>
#include <gphoto2/gphoto2.h>
#include <iostream>

using namespace Saiga;

#define CHECK_GP(_X)                                                                             \
    {                                                                                            \
        auto ret = _X;                                                                           \
        if (ret != GP_OK)                                                                        \
        {                                                                                        \
            std::cout << "Gphoto error in " << #_X << std::endl << "code: " << ret << std::endl; \
            SAIGA_ASSERT(0);                                                                     \
        }                                                                                        \
    }

class GphotoWrapper
{
   public:
    GphotoWrapper()
    {
        context = gp_context_new();
        CHECK_GP(gp_camera_new(&camera));
        CHECK_GP(gp_camera_init(camera, context));

        SAIGA_ASSERT(camera);

        CHECK_GP(gp_camera_get_config(camera, &config_widget, context));
        PrintWidget(config_widget);
    }

    ~GphotoWrapper()
    {
        gp_camera_exit(camera, context);
        gp_context_cancel(context);
    }

    std::vector<std::string> GetOptions(std::string name)
    {
        std::vector<std::string> result;

        SAIGA_ASSERT(config_map.count(name) > 0);
        auto w = config_map[name];
        SAIGA_ASSERT(w);

        CameraWidgetType widgetType;
        CHECK_GP(gp_widget_get_type(w, &widgetType));
        SAIGA_ASSERT(widgetType == CameraWidgetType::GP_WIDGET_RADIO);
        int cnt_choices = 0;
        cnt_choices     = gp_widget_count_choices(w);

        for (int i = 0; i < cnt_choices; ++i)
        {
            const char* choice;
            CHECK_GP(gp_widget_get_choice(w, i, &choice));
            result.push_back(choice);
        }
        return result;
    }

    void SetConfigValue(std::string name, std::string value)
    {
        SAIGA_ASSERT(config_map.count(name) > 0);
        auto w = config_map[name];
        SAIGA_ASSERT(w);
        CHECK_GP(gp_widget_set_value(w, value.c_str()));

        std::cout << "Set Camera Config " << name << " to " << value << std::endl;

        // CameraWidgetType widgetType;
        // CHECK_GP(gp_widget_get_type(w, &widgetType));
        // SAIGA_ASSERT(widgetType == CameraWidgetType::GP_WIDGET_RADIO);
        // int cnt_choices = 0;
        // cnt_choices     = gp_widget_count_choices(w);
        // std::cout << "choices " << cnt_choices << std::endl;
        //
        // for (int i = 0; i < cnt_choices; ++i)
        //{
        //    const char* choice;
        //    CHECK_GP(gp_widget_get_choice(w, i, &choice));
        //    CHECK_GP(gp_widget_set_value(w, choice));
        //    std::cout << "choice " << choice << std::endl;
        //}


        CHECK_GP(gp_camera_set_config(camera, config_widget, context));
    }

    void PrintWidget(CameraWidget* w, int d = 0)
    {
        const char* name;
        gp_widget_get_name(w, &name);

        auto c = gp_widget_count_children(w);
        std::cout << std::setw(d * 3) << "" << name << std::endl;

        if (c == 0)
        {
            config_map[name] = w;
        }

        for (int i = 0; i < c; ++i)
        {
            CameraWidget* child = nullptr;
            CHECK_GP(gp_widget_get_child(w, i, &child));
            PrintWidget(child, d + 1);
        }
    }



    std::vector<char> capture_to_memory()
    {
        std::vector<char> data;
        const char* ptr;
        unsigned long int size;
        CameraFile* file;
        CameraFilePath camera_file_path;

        /* NOP: This gets overridden in the library to /capt0000.jpg */
        strcpy(camera_file_path.folder, "/");
        strcpy(camera_file_path.name, "foo.jpg");

        CHECK_GP(gp_camera_capture(camera, GP_CAPTURE_IMAGE, &camera_file_path, context));

        CHECK_GP(gp_file_new(&file));

        CHECK_GP(gp_camera_file_get(camera, camera_file_path.folder, camera_file_path.name, GP_FILE_TYPE_NORMAL, file,
                                    context));

        CHECK_GP(gp_file_get_data_and_size(file, &ptr, &size));

        CHECK_GP(gp_camera_file_delete(camera, camera_file_path.folder, camera_file_path.name, context));

        data.resize(size);
        memcpy(data.data(), ptr, size);
        return data;
    }

    GPContext* context = nullptr;
    Camera* camera     = nullptr;
    CameraWidget* config_widget;
    std::map<std::string, CameraWidget*> config_map;
};

int main(int argc, char** argv)
{
    initSaigaSampleNoWindow();


    std::cout << "HDR Capture" << std::endl;

    GphotoWrapper camera;

    auto options = camera.GetOptions("shutterspeed");

    std::cout << "Available Shutter Speeds: " << std::endl;
    for (auto o : options) std::cout << o << std::endl;

    if (argc != 3)
    {
        std::cout << "Usage + example: " << std::endl;
        std::cout << "sample_vision_hdr_capture <start_exposure> <end_exposure>" << std::endl;
        std::cout << "sample_vision_hdr_capture 1/1000 2" << std::endl;
        return 0;
    }

    std::string start = argv[1];
    std::string end   = argv[2];

    auto it0 = std::find(options.begin(), options.end(), start);
    auto it1 = std::find(options.begin(), options.end(), end);

    if (distance(it0, it1) < 0)
    {
        std::swap(it0, it1);
    }
    ++it1;


    SAIGA_ASSERT(it0 != options.end());
    SAIGA_ASSERT(it1 != options.end());

    std::cout << "Capturing " << std::distance(it0, it1) << " photos..." << std::endl;


    for (auto it = it0; it != it1; ++it)
    {
        camera.SetConfigValue("shutterspeed", *it);

        auto data = camera.capture_to_memory();
        File::saveFileBinary(std::to_string(std::distance(it0, it)) + ".jpg", data.data(), data.size());
    }
}
