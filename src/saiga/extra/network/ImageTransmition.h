/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/image.h"

#include "boost/asio.hpp"

using namespace boost::asio;

namespace Saiga
{
class SAIGA_GLOBAL ImageTransmition
{
   public:
    static const uint32_t magicNumber     = 0x206a25f;
    static const uint32_t magicNumberData = 0x2b6a25f;

    struct ImageHeader
    {
        uint32_t mn = magicNumber;
        uint32_t width;
        uint32_t height;
        uint32_t pitch;
        uint32_t type;
        uint32_t numDataPackets;
        uint32_t imageId;
    };

    static const size_t headerSize = sizeof(ImageHeader);

    struct DataHeader
    {
        uint32_t mn = magicNumberData;
        uint32_t size;
        uint32_t offset;
        uint32_t imageId;
    };

    static constexpr size_t dataPacketSize = 1024 - sizeof(DataHeader);

    std::array<char, 10000> buffer;



    ImageTransmition(std::string host, uint32_t port);
    ~ImageTransmition();
    void makeReciever();
    void makeSender();


    // Create the transmition header for this image
    ImageHeader createHeader(const Image& img);
    // Check if the header fits to the image
    bool checkHeader(const Image& img, ImageHeader h);


    void sendHeader(const ImageHeader& h);
    bool recieveHeader(ImageHeader& img);

    size_t recievePacket();



    void sendImage(const Image& img);
    bool recieveImage(Image& img);
    bool recieveImageType(Image& img);

   private:
    io_service ios;
    boost::asio::ip::udp::socket socket;
    ip::udp::endpoint endpoint;
};

}  // namespace Saiga
