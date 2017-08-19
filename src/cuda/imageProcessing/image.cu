#include "saiga/cuda/imageProcessing/image.h"

namespace Saiga {
namespace CUDA {

void resizeDeviceVector(thrust::device_vector<uint8_t>& v, int size){
    v.resize(size);
}

}
}
