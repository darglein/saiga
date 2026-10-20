/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <filesystem>
#include <string>
#include <vector>

namespace Saiga
{
struct SAIGA_CORE_API ShaderCode
{
    std::vector<std::filesystem::path> dependent_files;

    std::vector<std::string> code;
    bool valid = false;

    struct Part
    {
        int start = 0, end = 0;
        std::string type = "";
    };

    std::vector<Part> parts;

    void DetectParts();
};

// This function loads the file given by a path and replaces #include commands
// by the respective code. All includes are
SAIGA_CORE_API ShaderCode LoadFileAndResolveIncludes(const std::filesystem::path& file, bool add_line_directives);

// Returns true if the line is an include directive: '#include "file"', '#include <file>' or '#include file', optionally
// followed by whitespace and a comment. 'file' is set to the included file, which is empty if it is malformed.
SAIGA_CORE_API bool ParseShaderInclude(const std::string& line, std::string& file);



}  // namespace Saiga
