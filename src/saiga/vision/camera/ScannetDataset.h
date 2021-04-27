/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/time/timer.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/camera/CameraBase.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"


namespace Saiga
{
// Scan-Net datasets from http://www.scan-net.org/
//
// The dataset must be extracted using their script into the following format.
//
// scene_0000_00
//      color
//          0.jpg
//          1.jpg
//          2.jpg
//          ...
//      depth
//          0.png
//          1.png
//          2.png
//          ...
//      pose
//          0.txt
//          1.txt
//          2.txt
//          ...
//      intrinsic
//          extrinsic_color.txt
//          extrinsic_depth.txt
//          intrinsic_color.txt
//          intrinsic_depth.txt
//
// After that you pass the absolute path to the 'scene_0000_00' directory in the DatasetParameters as 'dir'.
// The pose directory is optional and used as "ground truth" for error evaluation.
//
// Example:
//   DatasetParameters params;
//   params.dir = "/ssd/scannet/scene_0000_00/"
//
//   ScannetDataset dataset(params);
//   ...
//
class SAIGA_VISION_API ScannetDataset : public DatasetCameraBase
{
   public:
    ScannetDataset(const DatasetParameters& params, bool scale_down_color = true, bool scale_down_depth = true);
    virtual ~ScannetDataset() {}


    RGBDIntrinsics intrinsics() { return _intrinsics; }


    virtual void LoadImageData(FrameData& data) override;
    virtual int LoadMetaData() override;


   private:
    bool scale_down_color;
    bool scale_down_depth;
    RGBDIntrinsics _intrinsics;
    DMPP dmpp;
};

}  // namespace Saiga
