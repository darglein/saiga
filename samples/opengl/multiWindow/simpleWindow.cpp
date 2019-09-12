
#include "simpleWindow.h"

#include "saiga/opengl/shader/all.h"


int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();



    for (int i = 1; i <= 2; ++i)
    {
        {
            SampleWindowDeferred window;
            window.run();
        }


        std::cout << "window closed. Opening next window in..." << std::endl;

        for (int j = 0; i < 3 && j < 3; ++j)
        {
            std::cout << (3 - j) << std::endl;

            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }


    return 0;
}
