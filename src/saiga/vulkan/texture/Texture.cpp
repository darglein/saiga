#include "Texture.h"
#include "vkImageFormat.h"

namespace Saiga{
namespace Vulkan{

void Texture2D::fromImage(VulkanBase& base,Image &img)
{
    SAIGA_ASSERT(img.type == UC4);

    vk::Format format = getvkFormat(img.type);



    vk::FormatProperties formatProperties = base.physicalDevice.getFormatProperties(format);
//    vkGetPhysicalDeviceFormatProperties(device->physicalDevice, format, &formatProperties);


}

}
}
