#include "sample.h"
#include "saiga/framework.h"

int main(const int argc, const char *argv[])
{
    Saiga::SaigaParameters params;
    params.fromConfigFile("config.ini");
    Saiga::initSaiga(params);

    {
    VulkanExample vulkanExample;
    vulkanExample.setupWindow();
    vulkanExample.initVulkan();

    vulkanExample.init();
    vulkanExample.renderLoop();
    }

    Saiga::cleanupSaiga();
    return 0;
}
