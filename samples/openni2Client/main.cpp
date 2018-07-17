/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/openni2/RGBDCameraInput.h"

#include "boost/asio.hpp"
#include "saiga/util/ini/ini.h"

using namespace Saiga;

using namespace boost::asio;


void sendImage(
        TemplatedImage<ucvec4>& img,
        ip::udp::socket& socket,
        ip::udp::endpoint& remote_endpoint
        )
{
    size_t maxSize = 1024 * 4;
    size_t offset = 0;
    size_t size = img.size();

    while(offset < size)
    {
        size_t packetSize = std::min(maxSize,size-offset);
        auto buf = boost::asio::buffer(img.data8() + offset, packetSize);

    //        std::string str = "bla";
    //        auto buf = boost::asio::buffer(str.data(), str.size());
        auto size = socket.send_to(buf, remote_endpoint);
        cout << "send " << size << " bytes" << endl;

        offset += packetSize;
    }


}

int main(int argc, char *argv[])
{
    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("client","server_ip","10.0.0.2");
    auto port        = ini.GetAddLong ("client","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());


    boost::asio::io_service io_service;
    boost::asio::ip::udp::socket socket(io_service);
    socket.open(boost::asio::ip::udp::v4());

    ip::udp::resolver::query query(ip::udp::v4(),ip, std::to_string(port));
    ip::udp::resolver resolver(io_service);
    ip::udp::endpoint remote_endpoint = *resolver.resolve(query);
    cout << "address: " << remote_endpoint.address().to_string() << endl;




    boost::system::error_code err;


    RGBDCamera camera;
    camera.open();

    while(true)
    {
        camera.readFrame();

        sendImage(camera.colorImg,socket,remote_endpoint);

    }

    socket.close();
}
