#include "world/heightmap.h"
#include <libnoise/noise.h>
using namespace noise;


//typedef u_int32_t height_res_t;
//typedef u_int64_t height_resn_t;
//const int bits = 32;
//const height_res_t max_res = 4294967295;

typedef u_int16_t height_res_t;
typedef u_int32_t height_resn_t;
const int bits = 16;
const height_res_t max_res = 65535;

//typedef u_int8_t height_res_t;
//typedef u_int16_t height_resn_t;
//const int bits = 8;
//const height_res_t max_res = 255;

Heightmap::Heightmap(int layers, int w, int h):layers(layers),w(w),h(h){

    heights = new float[w*h];

    normalmap.resize(layers);



    heightmap.resize(layers);
    for(int i=0;i<layers;++i){
        heightmap[i].width = w;
        heightmap[i].height = h;
        heightmap[i].bitDepth = bits;
        heightmap[i].channels = 1;
        heightmap[i].create();

        normalmap[i].width = w;
        normalmap[i].height = h;
        normalmap[i].bitDepth = 8;
        normalmap[i].channels = 3;
        normalmap[i].create();

        w/=2;
        h/=2;
    }
}

void Heightmap::setScale(vec2 mapScale, vec2 mapOffset)
{
    this->mapOffset = mapOffset;
    this->mapScale = mapScale;
    this->mapScaleInv = 1.0f/mapScale;
}

void Heightmap::createInitialHeightmap(){




    PerlinNoise noise;

    module::RidgedMulti mountainTerrain;

    module::Perlin terrainType;
    terrainType.SetFrequency (0.5);
    terrainType.SetPersistence (0.25);

    module::Billow baseFlatTerrain;
      baseFlatTerrain.SetFrequency (2.0);
      module::ScaleBias flatTerrain;
        flatTerrain.SetSourceModule (0, baseFlatTerrain);
        flatTerrain.SetScale (0.125);
        flatTerrain.SetBias (-0.75);

        module::Select terrainSelector;
        terrainSelector.SetSourceModule (0, flatTerrain);
        terrainSelector.SetSourceModule (1, mountainTerrain);
        terrainSelector.SetControlModule (terrainType);
        terrainSelector.SetBounds (0.0, 1000.0);
        terrainSelector.SetEdgeFalloff (0.125);

        module::Turbulence finalTerrain;
        finalTerrain.SetSourceModule (0, terrainSelector);
        finalTerrain.SetFrequency (4.0);
        finalTerrain.SetPower (0.125);



    for(unsigned int x=0;x<w;++x){
        for(unsigned int y=0;y<h;++y){
            float xf = (float)x/(float)w;
            float yf = (float)y/(float)h;




            xf *= 10;
            yf *= 10;

            float wf = 10.0f;
            float hf = 10.0f;

//            double value = myModule.GetValue (xf, yf, 0);
//            float h = noise.fBm(xf,yf,0,5);
//            float h = myModule.GetValue (xf, yf, 0);
//            float h = finalTerrain.GetValue(xf,yf,0);

            //seamless
#define F(_X,_Y) finalTerrain.GetValue(_X,_Y,0)
            float he = (
            F(xf, yf) * (wf - xf) * (hf - yf) +
            F(xf - wf, yf) * (xf) * (hf - yf) +
            F(xf - wf, yf - hf) * (xf) * (yf) +
            F(xf, yf - hf) * (wf - xf) * (yf)
            ) / (wf * hf);

//            cout<<h<<endl;
            setHeight(x,y,he);

        }
    }

    normalizeHeightMap();

    for(unsigned int x=0;x<heightmap[0].width;++x){
        for(unsigned int y=0;y<heightmap[0].height;++y){

            float h = getHeight(x,y);
            h = h*max_res;
            h = glm::clamp(h,0.0f,(float)max_res);
            height_res_t n = (height_res_t)h;
            heightmap[0].setPixel(x,y,n);
        }
    }
}

void Heightmap::normalizeHeightMap(){
    float diff = maxH-minH;

    float m = minH;

    cout<<"min "<<minH<<" max "<<maxH<<endl;
    for(unsigned int x=0;x<w;++x){
        for(unsigned int y=0;y<h;++y){
            float h = getHeight(x,y);
            h = h-m;
            h = h/diff;
            setHeight(x,y,h);
        }
    }
}


void Heightmap::createNormalmap(){

    int layer = 0;
    for(int layer=0;layer<layers;++layer){
        for(int x=0;x<normalmap[layer].width;++x){
            for(int y=0;y<normalmap[layer].height;++y){

                vec3 norm(1.0f/w,1,1.0f/h);
//                vec3 scale = vec3(8*mapScaleInv.x,1,8*mapScaleInv.y) * norm;
//                 vec3 scale = vec3(200,1,200) * norm;
                 vec3 scale = vec3(mapScale.x,1,mapScale.y) * norm;


                vec3 x1 = vec3(x+1,getHeightScaled(layer,x+1,y),y) * scale;
                vec3 x2  = vec3(x-1,getHeightScaled(layer,x-1,y),y) * scale;

                vec3 y1  = vec3(x,getHeightScaled(layer,x,y+1),y+1) * scale;
                vec3 y2  = vec3(x,getHeightScaled(layer,x,y-1),y-1) * scale;

                vec3 n = glm::cross(y2-y1,x2-x1);

                //            cout<<"Normal "<<n<<" "<<x<<" "<<y<<" "<<x1<<" "<<x2<<" "<<y1<<" "<<y2<<endl;
                n = glm::normalize(n);
                n = 0.5f * (n+vec3(1.0f)); //now in range 0,1
                n = n*255.0f; //range 0,255
                n = glm::clamp(n,vec3(0),vec3(255));

                normalmap[layer].setPixel(x,y,(u_int8_t)n.x,(u_int8_t)n.y,(u_int8_t)n.z);

            }
        }
    }
}

float Heightmap::getHeight(int x, int y){
    x = glm::clamp(x,0,(int)(w-1));
    y = glm::clamp(y,0,(int)(h-1));

    return heights[x+y*w];
}

float Heightmap::getHeightScaled(int x, int y){
    return getHeight(x,y)*heightScale;
}

float Heightmap::getHeight(int layer, int x, int y){
    Image &img = heightmap[layer];

    x = glm::clamp(x,0,(int)(img.width-1));
    y = glm::clamp(y,0,(int)(img.height-1));

    height_res_t v = *((height_res_t*)img.positionPtr(x,y));
    return (float)v / (float)max_res;
}

float Heightmap::getHeightScaled(int layer, int x, int y){
    return getHeight(layer,x,y)*heightScale;
}

void Heightmap::setHeight(int x, int y, float v){
    minH = glm::min(v,minH);
    maxH = glm::max(v,maxH);
    heights[x+y*w] = v;
}

void Heightmap::createRemainingLayers(){
    for(int i=1;i<layers;++i){
        Image& previous = heightmap[i-1];
        Image& next = heightmap[i];
        //reduce previous to get the next
        cout<<"reduce next "<<next.width<<" "<<next.height<<endl;
        for(int x=0;x<next.width;++x){
            for(int y=0;y<next.height;++y){
                int xp = 2*x;
                int yp = 2*y;
                //read 4 pixel from previous and average them
                height_resn_t v1 = *((height_res_t*)previous.positionPtr(xp,yp));
                height_resn_t v2 = *((height_res_t*)previous.positionPtr(xp+1,yp));
                height_resn_t v3 = *((height_res_t*)previous.positionPtr(xp,yp+1));
                height_resn_t v4 = *((height_res_t*)previous.positionPtr(xp+1,yp+1));

                height_resn_t v = v1 + v2 + v3 + v4;
                v = (v / 4)+(v%4);
                next.setPixel(x,y,(height_res_t)v);
            }
        }
    }
}

void Heightmap::createTextures(){

    texheightmap.resize(layers);
    texnormalmap.resize(layers);

    PNG::Image img;

    for(int i=0;i<layers;i++){

//        heightmap[i].convertTo(img);
//        PNG::writePNG(&img,"heightmap"+std::to_string(i)+".png");

//        PNG::readPNG(&img,"heightmap"+std::to_string(i)+".png");
//        heightmap[i].convertFrom(img);

        texheightmap[i] = new Texture();
        texheightmap[i]->fromImage(heightmap[i]);
//        texheightmap[i]->setWrap(GL_CLAMP_TO_EDGE);
        texheightmap[i]->setWrap(GL_REPEAT);
        texheightmap[i]->setFiltering(GL_LINEAR);
//        texheightmap[i]->setFiltering(GL_NEAREST);

        texnormalmap[i] = new Texture();
        texnormalmap[i]->fromImage(normalmap[i]);
//        texnormalmap[i]->setWrap(GL_CLAMP_TO_EDGE);
        texnormalmap[i]->setWrap(GL_REPEAT);
        texnormalmap[i]->setFiltering(GL_LINEAR);
    }






}

void Heightmap::saveHeightmaps(){

    for(int i=0;i<layers;i++){
        fipImage fipimg;

        heightmap[i].convertTo(fipimg);

        string name = "heightmap"+std::to_string(i)+".png";

        if(fipimg.save(name.c_str())){
//            cout<<"save sucess"<<endl;
        }else{
            cout<<"save failed!"<<endl;
        }

    }
}
void Heightmap::saveNormalmaps(){


    for(int i=0;i<layers;i++){
        fipImage fipimg;

        normalmap[i].convertTo(fipimg);

        string name = "normalmap"+std::to_string(i)+".png";

        if(fipimg.save(name.c_str())){
//            cout<<"save sucess"<<endl;
        }else{
            cout<<"save failed!"<<endl;
        }


    }



}



bool Heightmap::loadMaps(){

    for(int i=0;i<layers;i++){
        string name = "heightmap"+std::to_string(i)+".png";

        fipImage fipimg;
        if (!fipimg.load(name.c_str()))
            return false;

        heightmap[i].convertFrom(fipimg);
    }

    for(int i=0;i<layers;i++){
        string name = "normalmap"+std::to_string(i)+".png";

        fipImage fipimg;
        if(!fipimg.load(name.c_str()))
            return false;

        normalmap[i].convertFrom(fipimg);
    }
    return true;
}


void Heightmap::createHeightmapsFrom(const string& image){
    fipImage fipimg;
    fipimg.load(image.c_str());
    heightmap[0].convertFrom(fipimg);


    createRemainingLayers();

    createNormalmap();

    saveHeightmaps();
    saveNormalmaps();

}


void Heightmap::createHeightmaps(){
    createInitialHeightmap();
    createRemainingLayers();

    createNormalmap();

    saveHeightmaps();
    saveNormalmaps();

}
