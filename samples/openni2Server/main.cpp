/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/openni2/RGBDCameraInput.h"
#include "boost/asio.hpp"
#include "saiga/util/ini/ini.h"
#include "saiga/network/ImageTransmition.h"

using namespace Saiga;
using namespace boost::asio;




int main(int argc, char *argv[]) {
    std::string file = "server.ini";
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());
    auto ip         = ini.GetAddString ("server","ip","10.0.0.2");
    auto port        = ini.GetAddLong ("server","port",9000);
    if(ini.changed()) ini.SaveFile(file.c_str());





    ImageTransmition it(ip,port);
    it.makeReciever();

    Image colorImg;
    while(true)
    {
        ip::udp::endpoint remote_endpoint;
        it.recieveImage(colorImg);
        cout << "recieved " << colorImg << endl;
    }

}
