#include "geometry/aabb.h"
#include <glm/gtc/epsilon.hpp>

aabb::aabb(void)
{
    min = glm::vec3(0,0,0);
    max = glm::vec3(0,0,0);
}

aabb::aabb(const glm::vec3 &p, const glm::vec3 &s) : min(p), max(s)
{
    //cout<<"init\n";
    //min = glm::vec3(p);
    //max = glm::vec3(s);
    //omin = glm::vec3(p);
    //omax = glm::vec3(s);
    //angleY = 0;

    //cout<<"init\n";
}

aabb::~aabb(void)
{
}

 void aabb::transform(const mat4 &trafo){
     //only for scaling and translation correct !!!!
     min = vec3(trafo*vec4(min,1));
     max = vec3(trafo*vec4(max,1));
 }

void aabb::makeNegative(){
    #define INFINITE 100000000000000.0f
    min = vec3(INFINITE);
    max = vec3(-INFINITE);
}
#define MIN(X,Y) ((X<Y)?X:Y)
#define MAX(X,Y) ((X>Y)?X:Y)
#define MINV(V1,V2) vec3(MIN(V1.x,V2.x),MIN(V1.y,V2.y),MIN(V1.z,V2.z))
#define MAXV(V1,V2) vec3(MAX(V1.x,V2.x),MAX(V1.y,V2.y),MAX(V1.z,V2.z))
void aabb::growBox(const vec3 &v){

    min = MINV(min,v);
    max = MAXV(max,v);
}

void aabb::growBox(const aabb &v){
    min = MINV(min,v.min);
    max = MAXV(max,v.max);
}


void aabb::translate(const glm::vec3 &v)
{
    min += v;
    max += v;

}

void aabb::scale(const glm::vec3 &s){
    vec3 pos = getPosition();
    setPosition(vec3(0));
    min*=s;
    max*=s;
    setPosition(pos);
}

vec3 aabb::getPosition() const
{
    return 0.5f*(min+max);

}

void aabb::setPosition(const glm::vec3 &v)
{
    vec3 mid = 0.5f*(min+max);
    mid = v-mid;
    translate(mid);

}


void aabb::ensureValidity()
{
    float tmp;
    if(min.x > max.x){
        tmp = min.x;
        min.x = max.x;
        max.x = tmp;
    }

    if(min.y > max.y){
        tmp = min.y;
        min.y = max.y;
        max.y = tmp;
    }

    if(min.z > max.z){
        tmp = min.z;
        min.z = max.z;
        max.z = tmp;
    }

}

int aabb::touching(const aabb &other){

    for(int i = 0;i<3;i++){
        //glm::equalEpsilon(1,1,1);
        if(glm::epsilonEqual(max[i], other.min[i],0.001f) && intersect2(other,i)) return 0x8<<i;
        if(glm::epsilonEqual(min[i], other.max[i],0.001f) && intersect2(other,i)) return 0x1<<i;
    }
    return -1;

}

bool aabb::intersect2(const aabb &other, int side){
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;
    side = (side+1)%3;
    if(min[side] >= other.max[side] || max[side] <= other.min[side] ) return false;

    return true; //overlap
}

int aabb::intersect(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return 0;
    if(min.y >= other.max.y || max.y <= other.min.y) return 0;
    if(min.z >= other.max.z || max.z <= other.min.z) return 0;

    if( other.min.x >= min.x && other.max.x <= max.x && //other inside this
            other.min.y >= min.y && other.max.y <= max.y &&
            other.min.z >= min.z && other.max.z <= max.z ) return 2; //contain
    return 1; //overlap
}

bool aabb::intersect2(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return false;
    if(min.y >= other.max.y || max.y <= other.min.y) return false;
    if(min.z >= other.max.z || max.z <= other.min.z) return false;

    return true; //overlap
}

bool aabb::intersectTouching(const aabb &other){
    if(min.x > other.max.x || max.x < other.min.x ) return false;
    if(min.y > other.max.y || max.y < other.min.y) return false;
    if(min.z > other.max.z || max.z < other.min.z) return false;

    return true; //overlap

}

int aabb::intersectAabb(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return 0;
    if(min.y >= other.max.y || max.y <= other.min.y) return 0;
    if(min.z >= other.max.z || max.z <= other.min.z) return 0;

    if( other.min.x >= min.x && other.max.x <= max.x && //other inside this
            other.min.y >= min.y && other.max.y <= max.y &&
            other.min.z >= min.z && other.max.z <= max.z ) return 2; //contain
    return 1; //overlap
}

bool aabb::intersectAabb2(const aabb &other){
    if(min.x >= other.max.x || max.x <= other.min.x ) return false;
    if(min.y >= other.max.y || max.y <= other.min.y) return false;
    if(min.z >= other.max.z || max.z <= other.min.z) return false;

    return true; //overlap
}



vec3 aabb::cornerPoint(int cornerIndex) const
{
    // assume(0 <= cornerIndex && cornerIndex <= 7);
    switch(cornerIndex)
    {
    default:
    case 0: return vec3(min.x, min.y, min.z);
    case 1: return vec3(min.x, min.y, max.z);
    case 2: return vec3(min.x, max.y, max.z);
    case 3: return vec3(min.x, max.y, min.z);
    case 4: return vec3(max.x, min.y, min.z);
    case 5: return vec3(max.x, min.y, max.z);
    case 6: return vec3(max.x, max.y, max.z);
    case 7: return vec3(max.x, max.y, min.z);
    }
}

bool aabb::contains(const glm::vec3 &p){
    if(min.x > p.x || max.x < p.x ) return false;
    if(min.y > p.y || max.y < p.y) return false;
    if(min.z > p.z || max.z < p.z) return false;

    return true; //overlap
}


//void aabb::getDrawData(GLfloat* vert, GLuint *vertpointer, GLuint *facedata, GLuint *facepointer, char visibility, int id){

//    static GLfloat tc[8][2] =
//    {
//        {1.0,0.0}, //bottom
//        {1.0,1.0},
//        {2.0,2.0},
//        {2.0,3.0}, //top

//        {1.0,4.0},
//        {1.0,5.0}, //bottom
//        {2.0,6.0}, //top
//        {2.0,7.0}
//    };


//    float l = 1/(glm::sqrt(3.0));
//    for(int i = 0;i<8;i++){
//        //x
//        int ind = (i*8)+(*vertpointer);
//        vert[ind+0] = (i&0x1) ? max.x : min.x;
//        vert[ind+3] = (i&0x1) ? l : -l;
//        //y
//        vert[ind+1] = (i&0x2) ? max.y : min.y;
//        vert[ind+4] = (i&0x2) ? l : -l;
//         //z
//        vert[ind+2] = (i&0x4) ? max.z : min.z;
//        vert[ind+5] = (i&0x4) ? l : -l;

//        //tex
//        vert[ind+6] = id;
//        vert[ind+7] = tc[i][1];


//    }


//    static GLuint ind[] = { 4,6,2, 2,0,4, //left
//                            4,0,1, 1,5,4, //bottom
//                            0,2,3, 3,1,0, //back
//                            1,3,7, 7,5,1, //right
//                            2,6,7, 7,3,2, //top
//                            5,7,6, 6,4,5}; //front



//    int count =0;
//    for(int i=0;i<6;i++){

//        if((visibility>>i)&0x1){

//            for(int j=0;j<6;j++){
//              facedata[count+j+(*facepointer)] = ind[i*6+j] + ((*vertpointer)/8);
//             }
//            count+=6;
//        }


//    }

//    *vertpointer += 8*8;
//    *facepointer += count;
//}

//void aabb::getDrawDataTx(GLfloat* vert, GLuint *vertpointer, GLuint *facedata, GLuint *facepointer, int visibility){

//    static GLfloat tc[8][2] =
//    {
//        {1.0,0.0}, //bottom
//        {1.0,1.0},
//        {2.0,2.0},
//        {2.0,3.0}, //top

//        {1.0,4.0},
//        {1.0,5.0}, //bottom
//        {2.0,6.0}, //top
//        {2.0,7.0}
//    };


//    float l = 1/(glm::sqrt(3.0));
//    for(int i = 0;i<8;i++){
//        //x
//        int ind = (i*8)+(*vertpointer);
//        vert[ind+0] = (i&0x1) ? max.x : min.x;
//        vert[ind+3] = (i&0x1) ? l : -l;
//        //y
//        vert[ind+1] = (i&0x2) ? max.y : min.y;
//        vert[ind+4] = (i&0x2) ? l : -l;
//         //z
//        vert[ind+2] = (i&0x4) ? max.z : min.z;
//        vert[ind+5] = (i&0x4) ? l : -l;

//        //tex
//        vert[ind+6] = tc[i][0];
//        vert[ind+7] = tc[i][1];


//    }



//    static GLuint ind[] = { 4,6,2, 2,0,4, //left
//                            4,0,1, 1,5,4, //bottom
//                            0,2,3, 3,1,0, //back
//                            1,3,7, 7,5,1, //right
//                            2,6,7, 7,3,2, //top
//                            5,7,6, 6,4,5}; //front



//    int count =0;
//    for(int i=0;i<6;i++){

//        if((visibility>>i)&1){

//            for(int j=0;j<6;j++){
//              facedata[count+j+(*facepointer)] = ind[i*6+j] + ((*vertpointer)/8);
//             }
//            count+=6;
//        }


//    }

//    *vertpointer += 8*8;
//    *facepointer += count;
//}

//void aabb::addOutlineToBuffer(std::vector<Vertex> &vertices,std::vector<GLuint> &indices){
//    int offset = vertices.size();
//    for(int i=0;i<8;i++){
//        vec3 c = cornerPoint(i);
//        Vertex v(c);
//        vertices.push_back(v);
//    }
//    static GLuint ind[] = { 0,1, 1,2, 2,3, 3,0,
//                                        4,5, 5,6, 6,7, 7,4,
//                                        0,4, 1,5, 2,6, 3,7
//                          };
//    for(int i=0;i<24;i++){
//        indices.push_back(ind[i]+offset);
//    }
//}

//void aabb::addToBuffer(std::vector<Vertex> &vertices,std::vector<GLuint> &indices){
//    int offset = vertices.size();
//    for(int i=0;i<8;i++){
//        vec3 c = cornerPoint(i);
//        Vertex v(c);
//        vertices.push_back(v);
//    }
//    static GLuint ind[] = { 4,6,2, 2,0,4, //left
//                            4,0,1, 1,5,4, //bottom
//                            0,2,3, 3,1,0, //back
//                            1,3,7, 7,5,1, //right
//                            2,6,7, 7,3,2, //top
//                            5,7,6, 6,4,5}; //front
//    for(int i=0;i<24;i++){
//        indices.push_back(ind[i+offset]);
//    }
//}

//void aabb::addToBuffer(std::vector<VertexN> &vertices,std::vector<GLuint> &indices){
//    int offset = vertices.size();

//    static vec3 normals[] = {
//        {-1,0,0},
//        {1,0,0},
//        {0,-1,0},
//        {0,1,0},
//        {0,0,-1},
//        {0,0,1}
//    };
//    static GLuint verts[] = {
//        0, 1, 2 ,3,
//        7, 6, 5, 4,
//        4, 5, 1 ,0,
//        3, 2, 6, 7,
//        0, 3, 7, 4,
//        2, 1, 5, 6
//    };

//    for(int f=0;f<6;f++){
//        for(int i=0;i<4;i++){
//            vec3 c = cornerPoint(verts[f*4+i]);
//            vec3 n= normals[f];
//            VertexN v(c,n);
//            vertices.push_back(v);
//        }

//    }
//    static GLuint ind[] = { 0,1,2, 2,3,0, //left
//                            4,5,6, 6,7,4, //bottom
//                            8,9,10, 10,11,8, //back
//                            12,13,14, 14,15,12, //right
//                            16,17,18, 18,19,16, //top
//                            20,21,22, 22,23,20 //front
//                          };
//    for(int i=0;i<36;i++){
//        indices.push_back(ind[i+offset]);
//    }
//}

std::ostream& operator<<(std::ostream& os, const aabb& bb)
{
    std::cout<<"aabb: ("<<bb.min.x<<","<<bb.min.y<<","<<bb.min.z<<")";
    std::cout<<" ("<<bb.max.x<<","<<bb.max.y<<","<<bb.max.z<<")";
    return os;
}
