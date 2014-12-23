#include "world/heightmap.h"

Heightmap::Heightmap(int layers, int w, int h):layers(layers),w(w),h(h){

    heights = new float[w*h];

    normalmap.resize(layers);



    heightmap.resize(layers);
    for(int i=0;i<layers;++i){
        heightmap[i].width = w;
        heightmap[i].height = h;
        heightmap[i].bitDepth = 16;
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

void Heightmap::createInitialHeightmap(){
    PerlinNoise noise;
    for(unsigned int x=0;x<w;++x){
        for(unsigned int y=0;y<h;++y){
            float xf = (float)x/(float)w;
            float yf = (float)y/(float)h;




            xf *= 10;
            yf *= 10;
            float h = noise.fBm(xf,yf,0,2);
//            cout<<h<<endl;
            setHeight(x,y,h);

        }
    }

    normalizeHeightMap();

    for(unsigned int x=0;x<heightmap[0].width;++x){
        for(unsigned int y=0;y<heightmap[0].height;++y){

            float h = getHeight(x,y);
            h = h*65536.0f;
            h = glm::clamp(h,0.0f,65535.0f);
            u_int16_t n = (u_int16_t)h;
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

    u_int16_t v = *((u_int16_t*)img.positionPtr(x,y));
    return (float)v / 65536.0f;
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
                u_int16_t v1 = *((u_int16_t*)previous.positionPtr(xp,yp));
                u_int16_t v2 = *((u_int16_t*)previous.positionPtr(xp+1,yp));
                u_int16_t v3 = *((u_int16_t*)previous.positionPtr(xp,yp+1));
                u_int16_t v4 = *((u_int16_t*)previous.positionPtr(xp+1,yp+1));

                u_int32_t v = v1 + v2 + v3 + v4;
                v = (v / 4)+(v%4);
                next.setPixel(x,y,(u_int16_t)v);
            }
        }
    }
}

void Heightmap::createTextures(){

    texheightmap.resize(7);
    texnormalmap.resize(7);

    for(int i=0;i<7;i++){
        texheightmap[i] = new Texture();
        texheightmap[i]->fromImage(heightmap[i]);
        texheightmap[i]->setWrap(GL_CLAMP_TO_EDGE);
        texheightmap[i]->setFiltering(GL_LINEAR);

        texnormalmap[i] = new Texture();
        texnormalmap[i]->fromImage(normalmap[i]);
        texnormalmap[i]->setWrap(GL_CLAMP_TO_EDGE);
    }
}

void Heightmap::createHeightmaps(){
    createInitialHeightmap();
    createRemainingLayers();



    createNormalmap();
}
