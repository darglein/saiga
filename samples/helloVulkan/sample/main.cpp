#include "sample.h"

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
    for (size_t i = 0; i < argc; i++) { VulkanExample::args.push_back(argv[i]); };
    vulkanExample = new VulkanExample();
    vulkanExample->initVulkan();
    vulkanExample->setupWindow();
    vulkanExample->prepare();
    vulkanExample->renderLoop();
    delete(vulkanExample);
    return 0;
}
