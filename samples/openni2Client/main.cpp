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



    //    boost::asio::ip::udp::socket socket(io_service);
    //    socket.open(boost::asio::ip::udp::v4());

    //    ip::udp::resolver::query query(ip::udp::v4(),ip, std::to_string(port));
    //    ip::udp::resolver resolver(io_service);
    //    ip::udp::endpoint remote_endpoint = *resolver.resolve(query);

    //    cout << "address: " << remote_endpoint.address().to_string() << endl;


    ImageTransmition it(ip,port);
    it.makeSender();


    boost::system::error_code err;


    RGBDCameraInput camera;
    RGBDCameraInput::CameraOptions co1;
    RGBDCameraInput::CameraOptions co2;
    co2.w = 320;
    co2.h = 240;
    camera.open( co1,co2);

    while(true)
    {
        camera.readFrame();

        it.sendImage(camera.colorImg);
        it.sendImage(camera.depthImg);
        cout << "image send" << endl;
        //        sendImage(camera.colorImg,socket,remote_endpoint);

    }
}
