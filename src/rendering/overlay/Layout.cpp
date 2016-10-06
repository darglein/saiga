#include "saiga/rendering/overlay/Layout.h"


Layout::Layout(int width, int height, float targetWidth, float targetHeight):width(width),height(height),targetWidth(targetWidth),targetHeight(targetHeight) {
    aspect = (float)width/(float)height;
    this->targetWidth = aspect;
    this->targetHeight = 1.0f;
    proj = glm::ortho(0.0f,this->targetWidth,0.0f,this->targetHeight,-1.0f,1.0f);
    scale = vec3(aspect,1.0f,1.0f);
}

aabb Layout::transform(Object3D *obj, const aabb &box, vec2 relPos, float relSize, Alignment alignmentX, Alignment alignmentY, bool scaleX)
{
    vec3 s = box.max-box.min;
    //scale to correct size
    if(scaleX){
        float ds = relSize/s.y;
        s = vec3(ds);
        s.x *= 1.0f/aspect;
    }else{
        float ds = relSize/s.x;
        s = vec3(ds);
        s.y *= 1.0f/aspect;
    }
    obj->scale = s;


    //alignment
    vec3 center = box.getPosition()*obj->scale;
    vec3 bbmin = box.min*obj->scale;
    vec3 bbmax = box.max*obj->scale;


    vec3 alignmentOffset(0);

    switch(alignmentX){
    case LEFT:
        alignmentOffset.x += bbmin.x;
        break;
    case RIGHT:
        alignmentOffset.x += bbmax.x;
        break;
    case CENTER:
        alignmentOffset.x += center.x;
        break;
    }

    switch(alignmentY){
    case LEFT:
        alignmentOffset.y += bbmin.y;
        break;
    case RIGHT:
        alignmentOffset.y += bbmax.y;
        break;
    case CENTER:
        alignmentOffset.y += center.y;
        break;
    }


    obj->position = vec3(relPos,0)-alignmentOffset;
//    cout << "obj position " << relPos << " " << alignmentOffset << " " << obj->position << endl;

    aabb resultBB = aabb(box.min*s,box.max*s);
    resultBB.setPosition(obj->position+center);

    obj->scale *= scale;
    obj->position  *= scale;

    obj->calculateModel();


    return resultBB;
}

aabb Layout::transformNonUniform(Object3D *obj, const aabb &box, vec2 relPos, vec2 relSize, Layout::Alignment alignmentX, Layout::Alignment alignmentY)
{
    vec3 s = box.max-box.min;
    s = vec3(relSize.x,relSize.y,1.0f) / vec3(s.x,s.y,1.0f);
    obj->scale = s;


    //alignment
    vec3 center = box.getPosition()*obj->scale;
    vec3 bbmin = box.min*obj->scale;
    vec3 bbmax = box.max*obj->scale;

    vec3 alignmentOffset(0);

    switch(alignmentX){
    case LEFT:
        alignmentOffset.x += bbmin.x;
        break;
    case RIGHT:
        alignmentOffset.x += bbmax.x;
        break;
    case CENTER:
        alignmentOffset.x += center.x;
        break;
    }

    switch(alignmentY){
    case LEFT:
        alignmentOffset.y += bbmin.y;
        break;
    case RIGHT:
        alignmentOffset.y += bbmax.y;
        break;
    case CENTER:
        alignmentOffset.y += center.y;
        break;
    }


    obj->position = vec3(relPos,0)-alignmentOffset;

    aabb resultBB = aabb(box.min*s,box.max*s);
    resultBB.setPosition(obj->position+center);

    obj->scale *= scale;
    obj->position  *= scale;
    obj->calculateModel();

    return resultBB;
}

aabb Layout::transformUniform(Object3D *obj, const aabb &box, vec2 relPos, vec2 relSize, Layout::Alignment alignmentX, Layout::Alignment alignmentY)
{
    relSize.x *= aspect;
    vec3 s = box.max-box.min;


//    s.x *= aspect;
    s = vec3(relSize.x,relSize.y,1.0f) / vec3(s.x,s.y,1.0f);

//    cout << "s: " << s << endl;
//    cout << "test: " << (s * (box.max-box.min)) << " " << relSize << endl;

    //use lower value of s.x and s.y to scale uniformly.
    //-> The result will fit in the box
    float ds = glm::min(s.x,s.y);

    obj->scale = vec3(ds,ds,1);
    obj->scale.x *= 1.0f/aspect;

    s = obj->scale;

    //alignment
    vec3 center = box.getPosition()*obj->scale;
    vec3 bbmin = box.min*obj->scale;
    vec3 bbmax = box.max*obj->scale;

    vec3 alignmentOffset(0);

    switch(alignmentX){
    case LEFT:
        alignmentOffset.x += bbmin.x;
        break;
    case RIGHT:
        alignmentOffset.x += bbmax.x;
        break;
    case CENTER:
        alignmentOffset.x += center.x;
        break;
    }

    switch(alignmentY){
    case LEFT:
        alignmentOffset.y += bbmin.y;
        break;
    case RIGHT:
        alignmentOffset.y += bbmax.y;
        break;
    case CENTER:
        alignmentOffset.y += center.y;
        break;
    }


    obj->position = vec3(relPos,0)-alignmentOffset;

    aabb resultBB = aabb(box.min*s,box.max*s);
    resultBB.setPosition(obj->position+center);

    obj->scale *= scale;
    obj->position  *= scale;
    obj->calculateModel();

    return resultBB;
}

glm::vec2 Layout::transformToLocal(glm::vec2 p)
{
    return vec2(p.x/width*targetWidth,p.y/height*targetHeight);
}

glm::vec2 Layout::transformToLocalNormalized(glm::vec2 p)
{
    return vec2(p.x/width,p.y/height);
}



