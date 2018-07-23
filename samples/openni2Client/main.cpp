/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/assert.h"
#include "saiga/openni2/RGBDCameraInput.h"

#include "saiga/network/ImageTransmition.h"
#include "boost/asio.hpp"
#include "saiga/util/ini/ini.h"


#include "saiga/time/timer.h"

#include <thread>
#include <condition_variable>
#include <mutex>

using namespace Saiga;

using namespace boost::asio;




int main(int argc, char *argv[])
{
    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("client","server_ip","10.0.0.2");
    auto port        = ini.GetAddLong ("client","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());


    ImageTransmition it(ip,port);
    it.makeSender();


    boost::system::error_code err;


    RGBDCameraInput camera;
    RGBDCameraInput::CameraOptions co1;
    RGBDCameraInput::CameraOptions co2;
    co2.w = 320;
    co2.h = 240;
    camera.open( co1,co2);


#if 0

    auto frame = camera.makeFrameData();

    int id = 0;

    AverageTimer at;
    for(int i =0; i < 300; ++i)
    {
        at.start();
        camera.readFrame(*frame);

        it.sendImage(frame->colorImg);
        it.sendImage(frame->depthImg);
        at.stop();


        cout << id << " image send. Fps: " << 1000.0 / at.getTimeMS() << endl;
        //        sendImage(camera.colorImg,socket,remote_endpoint);
        ++id;

    }
#else

    bool running = true;
    std::mutex lock1;
    std::condition_variable cv1;
     std::shared_ptr<RGBDCamera::FrameData> buffer1;

    std::thread pullThread(
                [&]()
    {
        std::shared_ptr<RGBDCamera::FrameData> frame = camera.makeFrameData();

        while(running)
        {
             camera.readFrame(*frame);
            {
                std::unique_lock<std::mutex> l(lock1);
                buffer1 = frame;
                cv1.notify_one();
            }
        }
    });

    int id = 0;

    AverageTimer at;
      for(int i =0; i < 300; ++i)
    {


          std::shared_ptr<RGBDCamera::FrameData> frame;
        {
            std::unique_lock<std::mutex> l(lock1);
            cv1.wait(l, [&](){return buffer1;}); //wait until the buffer is valid
            frame = buffer1;
            buffer1.reset();
        }



          it.sendImage(frame->colorImg);
          it.sendImage(frame->depthImg);
          at.stop();
          at.start();

          cout << id << " image send. Fps: " << 1000.0 / at.getTimeMS() << endl;
          //        sendImage(camera.colorImg,socket,remote_endpoint);
          ++id;
    }
      running = false;

    pullThread.join();

#endif
    return 0;
}
