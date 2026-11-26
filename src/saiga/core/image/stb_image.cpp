/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define STB_IMAGE_IMPLEMENTATION
#include "saiga/core/image/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#ifdef _MSC_VER
// stb (vendored) uses sprintf, which the MSVC CRT deprecates (C4996)
#    pragma warning(push)
#    pragma warning(disable : 4996)
#endif
#include "saiga/core/image/stb_image_write.h"
#ifdef _MSC_VER
#    pragma warning(pop)
#endif
