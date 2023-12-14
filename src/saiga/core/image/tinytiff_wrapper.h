#pragma once
#include "saiga/saiga_modules.h"

#ifdef SAIGA_USE_TINYTIFF

#include "saiga/core/image/managedImage.h"

namespace Saiga
{
	SAIGA_LOCAL bool loadImageTinyTiff(const std::string& path, Image& img);
}  // namespace Saiga

#endif
