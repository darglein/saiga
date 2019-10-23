/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "CameraData.h"

namespace Saiga
{
/**
 * Interface class for different datset inputs.
 */
template<typename FrameType>
class SAIGA_TEMPLATE CameraBase
{
   public:
    CameraBase() {}
    virtual ~CameraBase() {}

    // Blocks until the next image is available
    // Returns true if success.
    virtual bool getImageSync(FrameType& data) = 0;

    // Returns false if no image is currently available
    virtual bool getImage(FrameType& data) { return getImageSync(data); }

    virtual void close() {}
    virtual bool isOpened() { return true; }

   protected:
    int currentId = 0;
};

}  // namespace Saiga
