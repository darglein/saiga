#include "saiga/opengl/texture/imageMetaData.h"
#include "saiga/util/glm.h"

std::ostream& operator<<(std::ostream& os, const ImageMetadata& d){
    os << "> ImageMetadata" << endl;
    os << "Size: " << d.width << "x" << d.height << endl;
    os << "DateTime: " << d.DateTime << endl;
    os << "Make: " << d.Make << endl;
    os << "Model: " << d.Model << endl;

    os << "FocalLengthMM: " << d.FocalLengthMM << endl;

    std::string resStr;
    switch(d.FocalPlaneResolutionUnit){
    case ImageMetadata::NoUnit:
        resStr = "NoUnit";
        break;
    case ImageMetadata::Inch:
        resStr = "Inch";
        break;
    case ImageMetadata::Centimeter:
        resStr = "Centimeter";
        break;
    }

    os << "FocalPlaneResolutionUnit: " << resStr << endl;
    os << "FocalPlaneXResolution: " << d.FocalPlaneXResolution << endl;
    os << "FocalPlaneYResolution: " << d.FocalPlaneYResolution << endl;
    return os;
}

