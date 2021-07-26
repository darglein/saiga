/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#include <string>
#include <vector>

namespace Saiga
{
// Combined Command-line and config-file arguments
class SAIGA_CORE_API Arguments
{
   public:
    // Loads the given .ini file and sets the registered arguments
    void Load(const std::string& file, bool write_missing_values = true);

    void RegisterArgument(const std::string& name, float& f);


    struct Arg
    {
        std::string group;
        std::string name;
        std::string type;
        std::string value;
        int count = 0;
        void* data;
    };

    std::vector<Arg> args;
};

}  // namespace Saiga