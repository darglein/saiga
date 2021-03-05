/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#ifndef SAIGA_USE_SDL
#    error Saiga was compiled without SDL2.
#endif

#include <SDL.h>

#undef main
