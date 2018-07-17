/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/openni2/RGBDCameraInput.h"

#include "boost/asio.hpp"

using namespace Saiga;



int main(int argc, char *argv[])
{

    boost::asio::io_service io_service;
    boost::asio::ip::udp::socket socket(io_service);
    socket.open(boost::asio::ip::udp::v4());
    auto remote_endpoint = boost::asio::ip::udp::endpoint(boost::asio::ip::address::from_string("127.0.0.1"), 9000);





    boost::system::error_code err;


    RGBDCamera camera;
    camera.open();

    while(true)
    {
        camera.readFrame();

        auto buf = boost::asio::buffer("Jane Doe", 8);
        socket.send_to(buf, remote_endpoint, 0, err);



        cout << camera.depthImg(50,50) << " " << (int)camera.colorImg(50,50)[0] << endl;

    }

    socket.close();
}
