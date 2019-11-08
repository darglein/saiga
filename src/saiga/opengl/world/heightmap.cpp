/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/heightmap.h"

#include "saiga/config.h"
#include "saiga/opengl/texture/TextureLoader.h"

#ifdef USE_NOISE
#    include <libnoise/noise.h>
using namespace noise;
#endif

#undef min
#undef max


namespace Saiga
{
// typedef u_int32_t height_res_t;
// typedef u_int64_t height_resn_t;
// const int bits = 32;
// const height_res_t max_res = 4294967295;

typedef uint16_t height_res_t;
typedef uint32_t height_resn_t;
// const int bits             = 16;
const height_res_t max_res = 65535;

// typedef u_int8_t height_res_t;
// typedef u_int16_t height_resn_t;
// const int bits = 8;
// const height_res_t max_res = 255;

Heightmap::Heightmap(int layers, int w, int h) : layers(layers), w(w), h(h)
{
    heights = new float[w * h];

    normalmap.resize(layers);



    heightmap.resize(layers);
    for (int i = 0; i < layers; ++i)
    {
        heightmap[i].width  = w;
        heightmap[i].height = h;
        //        heightmap[i].Format() = ImageFormat(1,bits);
        SAIGA_ASSERT(0);
        //        heightmap[i].bitDepth = bits;
        //        heightmap[i].channels = 1;
        heightmap[i].create();

        normalmap[i].width  = w;
        normalmap[i].height = h;
        //        normalmap[i].bitDepth = 8;
        //        normalmap[i].channels = 3;
        //        normalmap[i].Format() = ImageFormat(3,8);
        SAIGA_ASSERT(0);
        normalmap[i].create();

        w /= 2;
        h /= 2;
    }
}

void Heightmap::setScale(vec2 mapScale, vec2 mapOffset)
{
#if 0
    this->mapOffset   = mapOffset;
    this->mapScale    = mapScale;
    this->mapScaleInv = 1.0f / mapScale;
#endif
}

void Heightmap::createInitialHeightmap()
{
    PerlinNoise noise;


#ifdef USE_NOISE
    module::RidgedMulti mountainTerrain;

    module::Perlin terrainType;
    terrainType.SetFrequency(0.3);
    terrainType.SetPersistence(0);

    module::Billow baseFlatTerrain;
    baseFlatTerrain.SetFrequency(2.0);
    module::ScaleBias flatTerrain;
    flatTerrain.SetSourceModule(0, baseFlatTerrain);
    flatTerrain.SetScale(0.125);
    flatTerrain.SetBias(-0.75);

    module::Select terrainSelector;
    terrainSelector.SetSourceModule(0, flatTerrain);
    terrainSelector.SetSourceModule(1, mountainTerrain);
    terrainSelector.SetControlModule(terrainType);
    terrainSelector.SetBounds(0.0, 1000.0);
    terrainSelector.SetEdgeFalloff(0.125);

    module::Turbulence finalTerrain;
    finalTerrain.SetSourceModule(0, terrainSelector);
    finalTerrain.SetFrequency(4.0);
    finalTerrain.SetPower(0.125);
#endif



    for (int x = 0; x < w; ++x)
    {
        for (int y = 0; y < h; ++y)
        {
            float xf = (float)x / (float)w;
            float yf = (float)y / (float)h;


            float f = 10.0f;

            xf *= f;
            yf *= f;

            float wf = f;
            float hf = f;

            //            double value = myModule.GetValue (xf, yf, 0);
            //            float h = noise.fBm(xf,yf,0,5);
            //            float h = myModule.GetValue (xf, yf, 0);
            //            float h = finalTerrain.GetValue(xf,yf,0);

            // seamless
#ifdef USE_NOISE
//#define F(_X,_Y) finalTerrain.GetValue(_X,_Y,0)
#    define F(_X, _Y) terrainType.GetValue(_X, _Y, 0)
#else
#    define F(_X, _Y) noise.fBm(_X, _Y, 0, 1)
#endif

            float he = (F(xf, yf) * (wf - xf) * (hf - yf) + F(xf - wf, yf) * (xf) * (hf - yf) +
                        F(xf - wf, yf - hf) * (xf) * (yf) + F(xf, yf - hf) * (wf - xf) * (yf)) /
                       (wf * hf);

            //            std::cout<<h<<endl;
            setHeight(x, y, he);
        }
    }

    normalizeHeightMap();

    for (int x = 0; x < heightmap[0].width; ++x)
    {
        for (int y = 0; y < heightmap[0].height; ++y)
        {
            float h = getHeight(x, y);
            h       = h * max_res;
            h       = clamp(h, 0.0f, (float)max_res);
            //            height_res_t n = (height_res_t)h;
            //            heightmap[0].setPixel(x,y,n);
            SAIGA_ASSERT(0);
        }
    }
}

void Heightmap::normalizeHeightMap()
{
    float diff = maxH - minH;

    float m = minH;

    std::cout << "min " << minH << " max " << maxH << std::endl;
    for (int x = 0; x < w; ++x)
    {
        for (int y = 0; y < h; ++y)
        {
            float h = getHeight(x, y);
            h       = h - m;
            h       = h / diff;
            setHeight(x, y, h);
        }
    }
}


void Heightmap::createNormalmap()
{
#if 0
    for (int layer = 0; layer < 1; ++layer)
    {
        for (int x = 0; x < normalmap[layer].width; ++x)
        {
            for (int y = 0; y < normalmap[layer].height; ++y)
            {
                vec3 norm(1.0f / w, 1, 1.0f / h);
                //                vec3 scale = vec3(8*mapScaleInv[0],1,8*mapScaleInv[1]) * norm;
                //                 vec3 scale = vec3(200,1,200) * norm;
                vec3 scale = ele_mult(vec3(mapScale[0], 1, mapScale[1]) , norm);
                //                vec3 scale = vec3(mapScaleInv[0],1,mapScaleInv[1]);


                //                vec3 x1 = vec3(x+1,getHeightScaled(layer,x+1,y),y) * scale;
                //                vec3 x2  = vec3(x-1,getHeightScaled(layer,x-1,y),y) * scale;
                //                vec3 y1  = vec3(x,getHeightScaled(layer,x,y+1),y+1) * scale;
                //                vec3 y2  = vec3(x,getHeightScaled(layer,x,y-1),y-1) * scale;

                vec3 x1 = vec3(x + 1, getHeightScaled(layer, x + 1, y), y) * scale;
                vec3 x2 = vec3(x - 1, getHeightScaled(layer, x - 1, y), y) * scale;

                vec3 y1 = vec3(x, getHeightScaled(layer, x, y + 1), y + 1) * scale;
                vec3 y2 = vec3(x, getHeightScaled(layer, x, y - 1), y - 1) * scale;

                //                float h1 = getHeightScaled(layer,x-1,y);
                //                float h2 = getHeightScaled(layer,x+1,y);
                //                float h3 = getHeightScaled(layer,x,y-1);
                //                float h4 = getHeightScaled(layer,x,y+1);

                //                vec2 step = vec2(1.0f,0.0f);
                //                vec3 va = normalize( vec3( 1, h2-h1,0 ));
                //                vec3 vb = normalize( vec3( 0, h4-h3,1 ));

                vec3 n = cross(y2 - y1, x2 - x1);

                std::swap(n[0], n[2]);
                //                vec3 n = cross(x2-x1,y2-y1);
                //                vec3 n = cross(vb,va);

                n = normalize(n);
                //                 std::cout<<"Normal "<<n<<endl;
                n = 0.5f * n + vec3(0.5f);  // now in range 0,1
                n = n * 255.0f;             // range 0,255
                //                std::cout<<"Normal "<<n<<endl;
                n = clamp(n, vec3(0), vec3(255));

                //                normalmap[layer].setPixel(x,y,(uint8_t)n[0],(uint8_t)n[1],(uint8_t)n[2]);
                SAIGA_ASSERT(0);
            }
        }
    }
#endif
}

float Heightmap::getHeight(int x, int y)
{
    x = clamp(x, 0, (int)(w - 1));
    y = clamp(y, 0, (int)(h - 1));

    return heights[x + y * w];
}

float Heightmap::getHeightScaled(int x, int y)
{
    return getHeight(x, y) * heightScale;
}

float Heightmap::getHeight(int layer, int x, int y)
{
    Image& img = heightmap[layer];


    while (x < 0) x += img.width;
    while (x >= img.width) x -= img.width;
    while (y < 0) y += img.height;
    while (y >= img.height) y -= img.height;


    //    x = clamp(x,0,(int)(img.width-1));
    //    y = clamp(y,0,(int)(img.height-1));

    //    height_res_t v = *((height_res_t*)img.positionPtr(x,y));
    //    return (float)v / (float)max_res;
    SAIGA_ASSERT(0);
    return 0;
}

float Heightmap::getHeightScaled(int layer, int x, int y)
{
    return getHeight(layer, x, y) * heightScale;
}

void Heightmap::setHeight(int x, int y, float v)
{
    minH               = std::min(v, minH);
    maxH               = std::max(v, maxH);
    heights[x + y * w] = v;
}

void Heightmap::createRemainingLayers()
{
    for (int i = 1; i < layers; ++i)
    {
        //        Image& previous = heightmap[i-1];
        Image& next = heightmap[i];
        // reduce previous to get the next
        std::cout << "reduce next " << next.width << " " << next.height << std::endl;
        for (int x = 0; x < next.width; ++x)
        {
            for (int y = 0; y < next.height; ++y)
            {
                //                int xp = 2*x;
                //                int yp = 2*y;
                // read 4 pixel from previous and average them
                SAIGA_ASSERT(0);
                //                height_resn_t v1 = *((height_res_t*)previous.positionPtr(xp,yp));
                //                height_resn_t v2 = *((height_res_t*)previous.positionPtr(xp+1,yp));
                //                height_resn_t v3 = *((height_res_t*)previous.positionPtr(xp,yp+1));
                //                height_resn_t v4 = *((height_res_t*)previous.positionPtr(xp+1,yp+1));

                //                height_resn_t v = v1 + v2 + v3 + v4;
                //                v = (v / 4)+(v%4);
                //                next.setPixel(x,y,(height_res_t)v);
            }
        }
    }
}

void Heightmap::createTextures()
{
    texheightmap.resize(layers);
    texnormalmap.resize(layers);

    //    PNG::Image img;

    for (int i = 0; i < layers; i++)
    {
        //        heightmap[i].convertTo(img);
        //        PNG::writePNG(&img,"heightmap"+std::to_string(i)+".png");

        //        PNG::readPNG(&img,"heightmap"+std::to_string(i)+".png");
        //        heightmap[i].convertFrom(img);

        //        texheightmap[i] = std::make_shared<Texture>();
        //        texheightmap[i]->fromImage(heightmap[i],false);
        //        texheightmap[i]->setWrap(GL_CLAMP_TO_EDGE);
        //        texheightmap[i]->setWrap(GL_REPEAT);
        //        texheightmap[i]->setFiltering(GL_LINEAR);
        //        texheightmap[i]->setFiltering(GL_NEAREST);

        //        texnormalmap[i] = std::make_shared<Texture>();
        //        texnormalmap[i]->fromImage(normalmap[i],false);
        //        texnormalmap[i]->setWrap(GL_CLAMP_TO_EDGE);
        //        texnormalmap[i]->setWrap(GL_REPEAT);
        //        texnormalmap[i]->setFiltering(GL_LINEAR);
    }
}

void Heightmap::saveHeightmaps()
{
    for (int i = 0; i < layers; i++)
    {
        std::string name = "heightmap" + std::to_string(i) + ".png";

        SAIGA_ASSERT(0);
        //        if(!TextureLoader::instance()->saveImage(name,heightmap[i])){
        //            std::cout<<"could not save "<<name<<endl;
        //        }
    }
}
void Heightmap::saveNormalmaps()
{
    for (int i = 0; i < layers; i++)
    {
        std::string name = "normalmap" + std::to_string(i) + ".png";


        //        if(!TextureLoader::instance()->saveImage(name,normalmap[i])){
        //            std::cout<<"could not save "<<name<<endl;
        //        }
    }
}



bool Heightmap::loadMaps()
{
    for (int i = 0; i < layers; i++)
    {
        std::string name = "heightmap" + std::to_string(i) + ".png";

        std::cout << "load heightmap " << std::endl;
        //        if (!TextureLoader::instance()->loadImage(name,heightmap[i]))
        //            return false;
    }

    for (int i = 0; i < layers; i++)
    {
        std::string name = "normalmap" + std::to_string(i) + ".png";

        //        if (!TextureLoader::instance()->loadImage(name,normalmap[i]))
        //            return false;
    }
    return true;
}


void Heightmap::createHeightmapsFrom(const std::string& image)
{
    //    if(!TextureLoader::instance()->loadImage(image,heightmap[0])){

    //    }


    createRemainingLayers();

    createNormalmap();

    saveHeightmaps();
    saveNormalmaps();
}


void Heightmap::createHeightmaps()
{
    createInitialHeightmap();
    createRemainingLayers();

    createNormalmap();

    saveHeightmaps();
    saveNormalmaps();
}

}  // namespace Saiga
