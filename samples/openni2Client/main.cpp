/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/openni2/RGBDCameraInput.h"

#include "boost/asio.hpp"
#include "saiga/util/ini/ini.h"

using namespace Saiga;



int main(int argc, char *argv[])
{
    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("server","ip","10.0.0.2");
    auto port        = ini.GetAddLong ("server","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());


    using namespace boost::asio;

    boost::asio::io_service io_service;
    boost::asio::ip::udp::socket socket(io_service);
    socket.open(boost::asio::ip::udp::v4());

    ip::udp::resolver::query query(ip::udp::v4(),ip, "1337");
    ip::udp::resolver resolver(io_service);
    ip::udp::endpoint remote_endpoint = *resolver.resolve(query);
    cout << "address: " << remote_endpoint.address().to_string() << endl;




    boost::system::error_code err;


    RGBDCamera camera;
//    camera.open();

    while(true)
    {
//        camera.readFrame();

        cout << "send" << endl;
        auto buf = boost::asio::buffer("Jane Doe", 8);
        socket.send_to(buf, remote_endpoint, 0, err);



//        cout << camera.depthImg(50,50) << " " << (int)camera.colorImg(50,50)[0] << endl;

    }

    socket.close();
}
