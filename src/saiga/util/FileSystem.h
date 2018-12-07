/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#if __has_include(<filesystem>)
#    include <filesystem>
#    define SAIGA_HAS_FILESYSTEM
#elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    define SAIGA_HAS_FILESYSTEM
#endif
