#include "sample.h"
#include "saiga/framework.h"
#include "saiga/vulkan/Shader/GLSL.h"
VulkanExample *vulkanExample;
//static void handleEvent(const xcb_generic_event_t *event)
//{
//    if (vulkanExample != NULL)
//    {
//        vulkanExample->handleEvent(event);
//    }
//}
int main(const int argc, const char *argv[])
{
    Saiga::SaigaParameters params;
    params.fromConfigFile("config.ini");
    Saiga::initSaiga(params);

    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->setupWindow();
    vulkanExample->initVulkan();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);

    Saiga::cleanupSaiga();
    return 0;
}
