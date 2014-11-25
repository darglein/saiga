#include "geometry/triangle.h"

void Triangle::stretch(){
    vec3 c = center();

    a = (a-c)*1.01f + a;
    b = (b-c)*1.01f + b;
    c = (c-c)*1.01f + c;
}

vec3 Triangle::center(){
    return (a+b+c)/vec3(3);
}

std::ostream& operator<<(std::ostream& os, const Triangle& t)
{
    std::cout<<"Triangle: "<<t.a<<t.b<<t.c;
    return os;
}
