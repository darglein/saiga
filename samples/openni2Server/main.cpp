/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/openni2/RGBDCameraInput.h"
#include "boost/asio.hpp"
#include "saiga/util/ini/ini.h"

using namespace Saiga;



int main(int argc, char *argv[]) {
    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("server","ip","10.0.0.2");
    auto port        = ini.GetAddLong ("server","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());


    using namespace boost::asio;

    io_service io_service;
    ip::udp::socket socket(io_service);
    socket.open(boost::asio::ip::udp::v4());

    ip::udp::resolver::query query(ip::udp::v4(),ip, std::to_string(port),ip::udp::resolver::query::canonical_name);
    ip::udp::resolver resolver(io_service);
    ip::udp::endpoint local_endpoint = *resolver.resolve(query);
    cout << "address: " << local_endpoint.address().to_string() << endl;

    socket.bind(local_endpoint);

    TemplatedImage<ucvec4> colorImg(480,640);
    while(true)
    {
        ip::udp::endpoint remote_endpoint;

//        auto buf = boost::asio::buffer(colorImg.data(), colorImg.size());
//        auto size = socket.receive_from(buf, remote_endpoint, 0);
//        cout << "recieved " << size << " bytes." << endl;





        std::array<char, 10000> recv_buf;

            size_t len = socket.receive_from(boost::asio::buffer(recv_buf), remote_endpoint);

//            std::cout.write(recv_buf.data(), len);
            std::cout << "recv " << len << endl;



//        cout << (int)colorImg(50,50)[0] << endl;

    }

    socket.close();
}
