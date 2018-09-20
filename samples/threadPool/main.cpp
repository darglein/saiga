/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <chrono>
#include "saiga/util/threadPool.h"

using namespace Saiga;


int main(int argc, char *argv[])
{

    ThreadPool tp(5);


    cout << "start" << endl;

    auto f = tp.enqueue([](){
        cout << "hello from other thread." << endl;
        std::this_thread::sleep_for(std::chrono::seconds(3));
    });


    tp.enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    tp.enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    tp.enqueue([](){
            cout << "hello from other thread." << endl;
            std::this_thread::sleep_for(std::chrono::seconds(3));
        });

    cout << "before wait " << endl;

    f.wait();

    cout << "Done." << endl;

}
