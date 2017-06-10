#include "saiga/rendering/particles/particleconverter.h"

#include "saiga/geometry/raytracer.h"

void Particleconverter::convert(std::vector<Triangle> &triangles, std::vector<vec3> &points){
    AABB box = getBoundingBox(triangles);
//    cout<<"box "<<box<<endl;

//    for(Triangle &t : triangles){
//        t.stretch();
//    }

    Raytracer rt(triangles);

    std::vector<Raytracer::Result> reslist;
    Ray r(vec3(-1,0,0),vec3(10,0,0));
    //    auto res = rt.trace(r);
    rt.trace(r,reslist);

//    cout<<"sorting..."<<endl;
    std::sort(reslist.begin(),reslist.end());





    vec3 particleSize(2);

    //number of samples in each direction
    vec3 size = box.max - box.min;
    int sx = ((int)glm::floor(size.x))/(int)particleSize.x+1;
    int sy = ((int)glm::floor(size.y))/(int)particleSize.y+1;
    int sz = ((int)glm::floor(size.z))/(int)particleSize.z+1;

//    cout<<"Samples "<<sx<<","<<sy<<","<<sz<<endl;


    vec3 s(sx-1,sy-1,sz-1);
//    cout<<"Range "<<s<<endl;

    vec3 start = ((size-(s*particleSize))/2.0f)+box.min;
    start.z = box.min.z - 1.0f;

//    std::vector<vec3> points; //list of points that are inside the object

//    Raytracer rt(triangles);

    for(int y=0;y<sy;++y){
        for(int x=0;x<sx;++x){
            vec3 pos = start+particleSize*vec3(x,y,0);
            Ray r(vec3(0,0,1),pos);
//            cout<<"Ray "<<r<<endl;

            reslist.resize(0);
            rt.trace(r,reslist);
            std::sort(reslist.begin(),reslist.end());

            bool foundstart = 0;
            decltype(reslist)::iterator start;
            for(auto it = reslist.begin();it!=reslist.end();++it){
                if(foundstart){
                    //find first back face
                    if((*it).back){
                        //voxelize range
                        voxelizeRange(points,r.getAlphaPosition((*start).distance),r.getAlphaPosition((*it).distance));
                        foundstart = false;
                    }
                }else{
                    //find first front face
                    if(!(*it).back){
                        start = it;
                        foundstart = true;
                    }
                }
            }
        }
    }

}

void Particleconverter::voxelizeRange(std::vector<vec3> &points,vec3 start, vec3 end){
    float distance = glm::distance(start,end);
    vec3 particleSize(2);
     int sx = ((int)glm::floor(distance))/(int)particleSize.x+1;

     float d = ((distance-((sx-1.0f)*particleSize.x))/2.0f);
     vec3 dir = glm::normalize(end-start);

     vec3 st = start + d*dir;

     for(int x=0;x<sx;++x){
         vec3 pos = st+particleSize.x*dir*(float)x;
          points.push_back(pos);
     }

//    points.push_back(start);
//    points.push_back(end);
}


AABB Particleconverter::getBoundingBox(std::vector<Triangle> &triangles){
    AABB box;
    box.makeNegative();
    for(Triangle &t : triangles){
        box.growBox(t.a);
        box.growBox(t.b);
        box.growBox(t.c);
    }
    return box;
}
