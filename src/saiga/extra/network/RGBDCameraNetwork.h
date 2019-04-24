/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#if 0

#    include "saiga/config.h"
#    include "saiga/core/image/image.h"

#    ifdef SAIGA_VISION

#        include "saiga/vision/RGBDCamera.h"



namespace Saiga
{
class ImageTransmition;

class SAIGA_EXTRA_API RGBDCameraNetwork : public RGBDCamera
{
   public:
    void connect(std::string host, uint32_t port);

    //    bool readFrame(FrameData& data) override;

    virtual std::shared_ptr<RGBDFrameData> waitForImage() override;
    virtual std::shared_ptr<RGBDFrameData> tryGetImage() override;

   private:
    std::shared_ptr<ImageTransmition> trans;
};

}  // namespace Saiga

#    endif

#endif
