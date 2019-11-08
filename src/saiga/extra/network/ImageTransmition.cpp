/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ImageTransmition.h"

#include "saiga/core/math/imath.h"

namespace Saiga
{
ImageTransmition::ImageTransmition(std::string host, uint32_t port) : socket(ios)
{
    socket.open(boost::asio::ip::udp::v4());
    ip::udp::resolver::query query(ip::udp::v4(), host, std::to_string(port));
    ip::udp::resolver resolver(ios);
    endpoint = *resolver.resolve(query);

    std::cout << "ImageTransmition " << endpoint.address().to_string() << " " << endpoint.port() << std::endl;
}

ImageTransmition::~ImageTransmition()
{
    socket.close();
}

void ImageTransmition::makeReciever()
{
    socket.bind(endpoint);
}

void ImageTransmition::makeSender() {}

ImageTransmition::ImageHeader ImageTransmition::createHeader(const Image& img)
{
    ImageHeader h;
    h.width          = img.width;
    h.height         = img.height;
    h.pitch          = img.pitchBytes;
    h.type           = img.type;
    h.numDataPackets = iDivUp(img.size(), dataPacketSize);
    h.imageId        = rand();
    return h;
}

bool ImageTransmition::checkHeader(const Image& img, ImageTransmition::ImageHeader h)
{
    return h.mn == magicNumber && (int)h.width == img.width && (int)h.height == img.height &&
           h.pitch == img.pitchBytes && (int)h.type == img.type;
}

void ImageTransmition::sendHeader(const ImageHeader& h)
{
    // first let's send a header with a magic number and the image properties
    //    std::array<uint32_t,5> header = { 0x206a25f, h.width, h.height, h.pitchBytes, h.type };
    //    size_t packetSize = header.size()*sizeof(uint32_t);
    auto buf      = boost::asio::buffer(&h, headerSize);
    auto sendSize = socket.send_to(buf, endpoint);
    SAIGA_ASSERT(sendSize == headerSize);
}

bool ImageTransmition::recieveHeader(ImageHeader& h)
{
    auto len = recievePacket();

    if (len != headerSize) return false;
    //    SAIGA_ASSERT(len == headerSize);

    memcpy(&h, buffer.data(), headerSize);

    if (h.mn != magicNumber) return false;

    return true;
}

size_t ImageTransmition::recievePacket()
{
    return socket.receive_from(boost::asio::buffer(buffer), endpoint);
}

void ImageTransmition::sendImage(const Image& img)
{
    ImageHeader h = createHeader(img);
    sendHeader(h);

    size_t imageSize = img.size();
    for (size_t i = 0; i < h.numDataPackets; ++i)
    {
        size_t offset     = i * dataPacketSize;
        size_t packetSize = std::min(dataPacketSize, imageSize - offset);

        DataHeader dh;
        dh.size    = packetSize;
        dh.offset  = offset;
        dh.imageId = h.imageId;

        const_buffers_1 bufHead = boost::asio::buffer((const void*)&dh, sizeof(DataHeader));
        const_buffers_1 bufData = boost::asio::buffer(img.data8() + offset, packetSize);


        std::array<const_buffers_1, 2> seq = {bufHead, bufData};

        auto size = socket.send_to(seq, endpoint);
        SAIGA_ASSERT(size == packetSize + sizeof(DataHeader));
    }
}

bool ImageTransmition::recieveImage(Image& img)
{
    ImageHeader h;

    // wait until the first header packet
    while (!recieveHeader(h))
    {
    }
    SAIGA_ASSERT(h.height > 0);


    while (true)
    {
        img.create(h.height, h.width, h.pitch, (Saiga::ImageType)h.type);
        SAIGA_ASSERT(checkHeader(img, h));

        //    std::cout << "got image" << std::endl;

        //    return;

        //        size_t imageSize = img.size();
        size_t i = 0;
        for (; i < h.numDataPackets; ++i)
        {
            recievePacket();

            DataHeader* dh = (DataHeader*)buffer.data();


            if (dh->mn != magicNumberData)
            {
                if (dh->mn == magicNumber)
                {
                    std::cout << "reset to next image " << dh->mn << std::endl;
                    memcpy(&h, buffer.data(), sizeof(ImageHeader));
                    //                    SAIGA_ASSERT(checkHeader(img,h));
                    i = 0;
                    break;
                }
                SAIGA_ASSERT(0);
                return false;
            }

            if (dh->imageId != h.imageId)
            {
                std::cout << "invalid imageid" << std::endl;
                i--;
                continue;
            }


            memcpy(img.data8() + dh->offset, buffer.data() + sizeof(DataHeader), dh->size);

            //        size_t offset = i * dataPacketSize;
            //        size_t packetSize = std::min(dataPacketSize,imageSize-offset);
            //        auto buf = boost::asio::buffer(img.data8() + offset, packetSize);
            //        auto size = socket.receive_from(buf, endpoint);
            //        std::cout << size << " " << packetSize << std::endl;
            //        SAIGA_ASSERT(size == packetSize);
        }

        if (i == h.numDataPackets) return true;
    }
    return false;
}

bool ImageTransmition::recieveImageType(Image& img)
{
    ImageHeader h;

    while (!recieveHeader(h))
    {
    }
    SAIGA_ASSERT(h.height > 0);

    if ((int)h.type != img.type) return false;

    SAIGA_ASSERT(checkHeader(img, h));


    size_t i = 0;
    for (; i < h.numDataPackets; ++i)
    {
        recievePacket();

        DataHeader* dh = (DataHeader*)buffer.data();


        if (dh->mn != magicNumberData)
        {
            return false;
        }

        if (dh->imageId != h.imageId)
        {
            return false;
        }
        memcpy(img.data8() + dh->offset, buffer.data() + sizeof(DataHeader), dh->size);
    }

    if (i == h.numDataPackets) return true;
    return false;
}

}  // namespace Saiga
