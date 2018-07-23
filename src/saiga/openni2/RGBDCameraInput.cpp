/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RGBDCameraInput.h"
#include "saiga/image/imageTransformations.h"

#include <OpenNI.h>
#include <thread>
#include "internal/noGraphicsAPI.h"


namespace Saiga {

bool RGBDCameraInput::open(CameraOptions rgbo, CameraOptions deptho)
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
        CameraOptions co =  deptho ;
        const openni::Array<openni::VideoMode>& modes = depth->getSensorInfo().getSupportedVideoModes();
        int found = -1;
        for(int i = 0; i < modes.getSize(); ++i)
        {
            const openni::VideoMode& mode = modes[i];
            //            cout << i <<  " supported mode: " << mode.getResolutionX() << "x" << mode.getResolutionY() << " " << mode.getFps() << " " << mode.getPixelFormat() << endl;

            if(mode.getResolutionX() == co.w &&
                    mode.getResolutionY() == co.h &&
                    mode.getFps() == co.fps &&
                    mode.getPixelFormat() == openni::PIXEL_FORMAT_DEPTH_1_MM
                    )
            {
                found = i;
                break;
            }
        }
        SAIGA_ASSERT(found != -1);
        auto rc = depth->setVideoMode(modes[found]);
        SAIGA_ASSERT(rc == openni::STATUS_OK);
    }

#endif
    cout << endl;
    {
         CameraOptions co =  rgbo ;
        const openni::Array<openni::VideoMode>& modes = color->getSensorInfo().getSupportedVideoModes();
          int found = -1;
        for(int i = 0; i < modes.getSize(); ++i)
        {
            const openni::VideoMode& mode = modes[i];
            //            cout << i <<  " supported mode: " << mode.getResolutionX() << "x" << mode.getResolutionY() << " " << mode.getFps() << " " << mode.getPixelFormat() << endl;

            if(mode.getResolutionX() == co.w &&
                    mode.getResolutionY() == co.h &&
                    mode.getFps() == co.fps &&
                    mode.getPixelFormat() == openni::PIXEL_FORMAT_RGB888
                    )
            {
                found = i;
                break;
            }
        }
          SAIGA_ASSERT(found != -1);
        auto rc = color->setVideoMode(modes[found]);
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


//    colorImg.create(colorH,colorW);
//    depthImg.create(depthH,depthW);


    cout << "RGBD Camera opened."  << endl;
    cout << "Color Resolution: " << colorW << "x" << colorH << endl;
    cout << "Depth Resolution: " << depthW << "x" << depthH << endl;
    return true;
}

bool RGBDCameraInput::readFrame(FrameData &data)
{
    openni::Status res;


    openni::VideoStream* streams[2] = {depth.get(),color.get()};
    int streamIndex;
    openni::OpenNI::waitForAnyStream(streams,2,&streamIndex);

    if(streamIndex == 0)
    {
//        std::thread t(&RGBDCameraInput::readDepth,this);
        readDepth(data.depthImg);
        readColor(data.colorImg);
//        t.join();

    }else{
        readColor(data.colorImg);
        readDepth(data.depthImg);


//        std::thread t(&RGBDCameraInput::readColor,this);

//        readDepth();
//        t.join();
    }






    return true;
}

bool RGBDCameraInput::readDepth(ImageView<unsigned short> depthImg)
{
    auto res = depth->readFrame(m_depthFrame.get());
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
    return true;
}

bool RGBDCameraInput::readColor(ImageView<ucvec4> colorImg)
{
    auto res = color->readFrame(m_colorFrame.get());
    if (res != openni::STATUS_OK) return false;

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
