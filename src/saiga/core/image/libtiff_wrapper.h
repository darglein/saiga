#pragma once
#include "saiga/saiga_modules.h"

#ifdef SAIGA_USE_LIBTIFF

#include "saiga/core/image/managedImage.h"

namespace Saiga
{
	SAIGA_LOCAL bool loadImageLibTiff(const std::string& path, Image& img);
}  // namespace Saiga

#endif
