/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <thrust/version.h>

#if THRUST_VERSION < 100700
#    error "Thrust v1.7.0 or newer is required"
#endif

#include <thrust/detail/config.h>

#include <thrust/detail/config/host_device.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
