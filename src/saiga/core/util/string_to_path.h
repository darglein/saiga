#pragma once

#include "saiga/config.h"
#include <filesystem>

SAIGA_CORE_API bool is_valid_utf8(const std::string& string);
SAIGA_CORE_API std::filesystem::path string_to_path(const std::string& s);
SAIGA_CORE_API std::string path_to_string(const std::filesystem::path& s);

