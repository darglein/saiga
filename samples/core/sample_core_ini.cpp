/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"


using namespace Saiga;


struct SampleParams : public ParamsBase
{
    SAIGA_PARAM_STRUCT_FUNCTIONS(SampleParams);
    virtual void Params(Saiga::SimpleIni* ini, CLI::App* app) override
    {
        SAIGA_PARAM_COMMENT(split_method, "more comments");
        SAIGA_PARAM(max_images);
        SAIGA_PARAM_COMMENT(train_on_eval, "test comment");
        SAIGA_PARAM(train_factor);
    }

    std::string split_method = "";
    bool train_on_eval       = false;
    double train_factor      = 0.9;
    int max_images           = -1;
};

int main(int argc, char* argv[])
{

    SampleParams params("params.ini");

    CLI::App app{"Example programm", "exmaple_programm"};
    params.Load(app);
    CLI11_PARSE(app, argc, argv);
    std::cout << "split:" <<  params.split_method << std::endl;

    /**
     * This sample demonstrates the use of the ini class.
     */
    auto fileName = "example.ini";

    Saiga::Ini ini;
    ini.LoadFile(fileName);


    std::string name;
    int w, h;
    bool b;
    mat4 m  = mat4::Identity();
    m(0, 1) = 1;  // row 0 and col 1

    name = ini.GetAddString("window", "name", "Test Window");
    w    = ini.GetAddLong("window", "width", 1280);
    h    = ini.GetAddDouble("window", "height", 720);
    b    = ini.GetAddBool("window", "fullscreen", false);
    Saiga::fromIniString(ini.GetAddString("window", "viewmatrix", Saiga::toIniString(m).c_str()), m);

    std::cout << name << " " << w << "x" << h << " " << b << " " << std::endl << m << std::endl;

    if (ini.changed()) ini.SaveFile(fileName);


}
