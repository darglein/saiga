/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "model_loader_off.h"

#include "saiga/core/math/String.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>

namespace Saiga
{
static StringViewParser lineParser = {" ", true};
static StringViewParser faceParser = {"/", false};



OffModelLoader::OffModelLoader(const std::string& file) : file(file)
{
    loadFile(file);
}


bool OffModelLoader::loadFile(const std::string& _file)
{
    this->file = SearchPathes::model(_file);
    if (file == "")
    {
        std::cerr << "Could not open file " << _file << std::endl;
        std::cerr << SearchPathes::model << std::endl;
        return false;
    }

    std::cout << "[OffModelLoader] Loading " << file << std::endl;

    std::vector<std::string> data = File::loadFileStringArray(file);
    File::removeWindowsLineEnding(data);

    int num_vertices = 0, num_faces = 0;

    for (auto& line : data)
    {
        if (state == ParsingState::HEADER)
        {
            if (line == "OFF")
            {
                state = ParsingState::COUNTS;
            }
        }
        else if (state == ParsingState::COUNTS)
        {
            StringViewParser lineParser = {" ", true};
            lineParser.set(line);

            num_vertices = to_long(lineParser.next());
            num_faces    = to_long(lineParser.next());


            mesh.vertices.reserve(num_vertices);
            mesh.faces.reserve(num_faces);

            state = ParsingState::VERTICES;
        }
        else if (state == ParsingState::VERTICES)
        {
            StringViewParser lineParser = {" ", true};
            lineParser.set(line);

            vec3 p;
            p[0] = to_double(lineParser.next());
            p[1] = to_double(lineParser.next());
            p[2] = to_double(lineParser.next());

            VertexNC v;
            v.position = make_vec4(p, 1);
            v.color    = make_vec4(1);

            mesh.vertices.push_back(v);

            if ((int)mesh.vertices.size() == num_vertices)
            {
                state = ParsingState::FACES;
            }
        }
        else if (state == ParsingState::FACES)
        {
            StringViewParser lineParser = {" ", true};
            lineParser.set(line);
            int count = to_long(lineParser.next());

            SAIGA_ASSERT(count == 3 || count == 4);
            std::array<uint32_t, 4> indices;
            for (int i = 0; i < count; ++i)
            {
                indices[i] = to_long(lineParser.next());
            }

            if (count == 3)
            {
                mesh.addFace(indices.data());
            }
            else if (count == 4)
            {
                mesh.addQuad(indices.data());
            }


            if ((int)mesh.faces.size() == num_faces)
            {
                state = ParsingState::DONE;
            }
        }
    }

    mesh.computePerVertexNormal();

    if (state != ParsingState::DONE)
    {
        std::cout << "Parsing failed!" << std::endl;
        return false;
    }


    std::cout << "[OffModelLoader] Done. " << mesh << std::endl;

    return true;
}



}  // namespace Saiga
