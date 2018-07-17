/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCameraInput.h"
#include "saiga/image/imageTransformations.h"

#include <OpenNI.h>


namespace Saiga {

bool RGBDCamera::open()
{

    device = std::make_shared<openni::Device>();
    depth = std::make_shared<openni::VideoStream>();
    color = std::make_shared<openni::VideoStream>();
    m_depthFrame = std::make_shared<openni::VideoFrameRef>();
    m_colorFrame = std::make_shared<openni::VideoFrameRef>();



    cout << "rgbd camera open." << endl;


    openni::Status rc = openni::STATUS_OK;


    const char* deviceURI = openni::ANY_DEVICE;

    rc = openni::OpenNI::initialize();

    printf("After initialization:\n%s\n", openni::OpenNI::getExtendedError());


    openni::Array<openni::DeviceInfo> deviceInfoList;
    openni::OpenNI::enumerateDevices(&deviceInfoList);



    for(int i = 0; i < deviceInfoList.getSize(); ++i)
    {
        auto di = deviceInfoList[i];
        cout << di.getName() << " " << di.getUri() << " " << di.getUsbProductId() << " " << di.getUsbVendorId() << " " << di.getVendor() << endl;
    }



    deviceURI = deviceInfoList[0].getUri();

    rc = device->open(deviceURI);
    if (rc != openni::STATUS_OK)
    {
        printf("SimpleViewer: Device open failed:\n%s\n", openni::OpenNI::getExtendedError());
        openni::OpenNI::shutdown();
        return false;
    }

#if 1
    rc = depth->create(*device, openni::SENSOR_DEPTH);
    if (rc == openni::STATUS_OK)
    {

    }
    else
    {
        printf("SimpleViewer: Couldn't find depth stream:\n%s\n", openni::OpenNI::getExtendedError());
    }



#endif
    rc = color->create(*device, openni::SENSOR_COLOR);
    if (rc == openni::STATUS_OK)
    {

    }
    else
    {
        printf("SimpleViewer: Couldn't find color stream:\n%s\n", openni::OpenNI::getExtendedError());
    }

#if 1
    if (!depth->isValid() || !color->isValid())
    {
        printf("SimpleViewer: No valid streams. Exiting\n");
        openni::OpenNI::shutdown();
        return false;
    }
    {
        const openni::Array<openni::VideoMode>& modes = depth->getSensorInfo().getSupportedVideoModes();
        for(int i = 0; i < modes.getSize(); ++i)
        {
            const openni::VideoMode& mode = modes[i];
            cout << i <<  " supported mode: " << mode.getResolutionX() << "x" << mode.getResolutionY() << " " << mode.getFps() << " " << mode.getPixelFormat() << endl;
        }
        auto rc = depth->setVideoMode(modes[4]);
        SAIGA_ASSERT(rc == openni::STATUS_OK);
    }

#endif
    cout << endl;
    {
        const openni::Array<openni::VideoMode>& modes = color->getSensorInfo().getSupportedVideoModes();
        for(int i = 0; i < modes.getSize(); ++i)
        {
            const openni::VideoMode& mode = modes[i];
            cout << i <<  " supported mode: " << mode.getResolutionX() << "x" << mode.getResolutionY() << " " << mode.getFps() << " " << mode.getPixelFormat() << endl;
        }
        auto rc = color->setVideoMode(modes[9]);
        SAIGA_ASSERT(rc == openni::STATUS_OK);
    }

    cout << endl;

    rc = color->start();
    if (rc != openni::STATUS_OK)
    {
        printf("SimpleViewer: Couldn't start color stream:\n%s\n", openni::OpenNI::getExtendedError());
        color->destroy();
    }


    rc = depth->start();
    if (rc != openni::STATUS_OK)
    {
        printf("SimpleViewer: Couldn't start depth stream:\n%s\n", openni::OpenNI::getExtendedError());
        depth->destroy();
    }

    int m_width;
    int m_height;

    openni::VideoMode depthVideoMode;
    openni::VideoMode colorVideoMode;


    colorW = color->getVideoMode().getResolutionX();
    colorH = color->getVideoMode().getResolutionY();

    depthW = depth->getVideoMode().getResolutionX();
    depthH = depth->getVideoMode().getResolutionY();


    colorImg.create(colorH,colorW);
    depthImg.create(depthH,depthW);


    cout << "RGBD Camera opened."  << endl;
    cout << "Color Resolution: " << colorW << "x" << colorH << endl;
    cout << "Depth Resolution: " << depthW << "x" << depthH << endl;
    return true;
}

bool RGBDCamera::readFrame()
{
    openni::Status res;

    res = depth->readFrame(m_depthFrame.get());
    if (res != openni::STATUS_OK) return false;

    res = color->readFrame(m_colorFrame.get());
    if (res != openni::STATUS_OK) return false;

    ImageView<uint16_t> rawDepthImg(
                m_depthFrame->getHeight(),
                m_depthFrame->getWidth(),
                m_depthFrame->getStrideInBytes(),
                (void*)m_depthFrame->getData());
    for(int i = 0; i < rawDepthImg.height; ++i)
    {
        for(int j =0; j < rawDepthImg.width; ++j)
        {
            depthImg(i,j) = rawDepthImg(i,rawDepthImg.width-j-1);
        }
    }


    ImageView<ucvec3> rawImg(
                m_colorFrame->getHeight(),
                m_colorFrame->getWidth(),
                m_colorFrame->getStrideInBytes(),
                (void*)m_colorFrame->getData());
    for(int i = 0; i < rawImg.height; ++i)
    {
        for(int j =0; j < rawImg.width; ++j)
        {
            colorImg(i,j) = ucvec4(rawImg(i,rawImg.width-j-1),0);
        }
    }

    return true;
}


}
