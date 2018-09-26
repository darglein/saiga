/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <chrono>
#include "saiga/util/threadPool.h"
#include "saiga/util/pipeline.h"

using namespace Saiga;

struct Bla{
    int foo[50];
};


Saiga::PipelineStage<std::shared_ptr<Bla>,5,true> pipeline;

void startPipeline()
{
    pipeline.run([&]()
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        return std::make_shared<Bla>();
    });
}

void testRingBuffer()
{
    startPipeline();

    for(int i = 0; i < 1000; ++i)
    {
        std::shared_ptr<Bla> a;
        auto v = pipeline.tryGet(a);
        if(v)
        {
            for(int i = 0; i < 10; ++i)
            {
                a->foo[i] = 5;
            }
        }


        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }

    pipeline.stop();
}

int main(int argc, char *argv[])
{

    SynchronizedBuffer<int> buffer(5);
//    std::vector<char> asdf(100);

//    testRingBuffer();
    return 0;

    createGlobalThreadPool(5);



    cout << "start" << endl;

    auto f = globalThreadPool->enqueue([](){
        cout << "hello from other thread." << endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    });


    globalThreadPool->enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    globalThreadPool->enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    globalThreadPool->enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    cout << "before wait " << endl;

    f.wait();

    cout << "Done." << endl;

}
