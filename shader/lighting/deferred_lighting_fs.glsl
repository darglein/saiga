uniform mat4 model;
uniform mat4 view;
uniform mat4 proj;

uniform mat4 invProj;

//converts camera view space -> shadow clip space
uniform mat4 depthBiasMV;
//uniform sampler2DShadow depthTex;



uniform sampler2D deferred_diffuse;
uniform sampler2D deferred_normal;
uniform sampler2D deferred_depth;
uniform sampler2D deferred_position;
uniform sampler2D deferred_data;
uniform vec2 screen_size;

uniform vec4 color;




vec2 CalcTexCoord()
{
   return gl_FragCoord.xy / screen_size;
}



float intensityDiffuse(vec3 normal, vec3 lightDir){
    return max(dot(normal,lightDir), 0.0);
}

float intensitySpecular(vec3 position, vec3 normal, vec3 lightDir, float exponent){
    vec3 E = normalize(-position);
    vec3 R = normalize(-reflect(lightDir,normal));
    float i = pow(max(dot(R,E),0.0),exponent);
    return clamp(i,0.0f,1.0f);
}


float getFadeOut(float d, float r){
    //makes sure that the light at the border of the point light is always 0, with a smooth transition
    //not physically corrrect, but looks good


    //the light intensity will smoothly drecrease from b->xs
    const float b = 0.6;
    const float xs = 0.9f;


    const float m = -(1/(xs-b));
    const float t = -m*xs;

    float alpha = d/r;
    float x = m * alpha + t;
    return clamp(x,0,1);
}

float getAttenuation(vec3 attenuation, float distance, float radius){
//    float d = distance(fragposition,lightPos);


    return getFadeOut(distance,radius) / (attenuation.x +
                    attenuation.y * distance +
                    attenuation.z * distance * distance);
}

float calculateShadow(sampler2DShadow tex, vec3 position, float outside){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;

     float visibility = 1.0f;
     if ((shadowPos.x < 0 || shadowPos.x > 1 || shadowPos.y < 0 || shadowPos.y > 1 || shadowPos.z < 0 || shadowPos.z > 1)){
         visibility = outside;
     }else{
         //the bias is applied with glPolygonOffset
         visibility = texture(tex, shadowPos.xyz);
     }
     return visibility;

}

float calculateShadow(sampler2DShadow tex, vec3 position){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;

     float visibility = 1.0f;
         visibility = texture(tex, shadowPos.xyz);
     return visibility ;

}

float calculateShadowPCF(sampler2D tex, vec3 position){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;

     float bias =  0.0012;
     float visibility = 1.0f;
//    visibility = texture(depthTex, vec3(shadowPos.xy, shadowPos.z -bias));
     if ((shadowPos.x < 0 || shadowPos.x > 1 || shadowPos.y < 0 || shadowPos.y > 1 || shadowPos.z < 0 || shadowPos.z > 1)){
         visibility = 1.0f;
     }else{
        visibility = 0.0f;
         float pcfSize = 1;
         float samples = (pcfSize+1.0)*2.0 * (pcfSize+1.0)*2.0;
        const float texScale = 1.0/512.0;

         for(float u = -pcfSize ; u <= pcfSize ; u = u+1.0){
             for(float v = -pcfSize ; v <= pcfSize ; v = v+1.0){
                 vec2 offset = vec2(u,v)*texScale;
                 float shadowDepth = 0;
                 shadowDepth = texture(tex, shadowPos.xy+offset).r;
                 visibility += (shadowDepth<shadowPos.z-bias) ? 0.0f : 1.0f;
             }
         }
         visibility *= 1.0/(samples);
     }
     return visibility;

}


float VectorToDepth (vec3 Vec, float farplane, float nearplane)
{
    vec3 AbsVec = abs(Vec);
    float LocalZcomp = max(AbsVec.x, max(AbsVec.y, AbsVec.z));

    // Replace f and n with the far and near plane values you used when
    //   you drew your cube map.
    float f = farplane;
    float n = nearplane;

    float NormZComp = (f+n) / (f-n) - (2*f*n)/(f-n)/LocalZcomp;
    return (NormZComp + 1.0) * 0.5;
}

float calculateShadowCube(samplerCubeShadow tex, vec3 lightW, vec3 fragW, float farplane, float nearplane){
    vec3 direction =  fragW-lightW;
    float visibility = 1.0f;

    float d = VectorToDepth(direction,farplane,nearplane);
    //the bias is applied with glPolygonOffset
    visibility = texture(tex, vec4(direction,d));
    return visibility;
}

vec3 unpackNormal2 (vec2 enc)
{
    vec3 n;
    n.z=length(enc)*2-1;

    n.xy= normalize(enc)*sqrt(1-n.z*n.z);
    return n;
}

vec3 unpackNormal3 (vec2 enc)
{
    vec2 fenc = enc*4-vec2(2);
    float f = dot(fenc,fenc);
    float g = sqrt(1-f/4);
    vec3 n;
    n.xy = fenc*g;
    n.z = 1-f/2;
    return n;
}


vec3 reconstructPosition(float d, vec2 tc){

//    float x = tc.x * 2 - 1;
//    float y = (tc.y) * 2 - 1;
//    float z = d*2-1;

    vec4 p = vec4(tc.x,tc.y,d,1)*2.0f - 1.0f;

    p = invProj * p;
    return p.xyz/p.w;
//    return vec3(x,y,1);
}

//void getGbufferData(out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data){

//    vec2 tc = CalcTexCoord();
//    color = texture( deferred_diffuse, tc ).rgb;
//    position = texture( deferred_position, tc ).xyz;
//    depth = texture( deferred_depth, tc ).r;
//    normal = texture( deferred_normal, tc ).xyz;
//    normal = normal*2.0f - 1.0f;
//    data = texture(deferred_data,tc).xyz;
//}


void getGbufferData(out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data){

    vec2 tc = CalcTexCoord();
    color = texture( deferred_diffuse, tc ).rgb;

//    position = texture( deferred_position, tc ).xyz;

    depth = texture( deferred_depth, tc ).r;
    position = reconstructPosition(depth,tc);

    normal = texture( deferred_normal, tc ).xyz;
//    normal = normal*2.0f - 1.0f;
    normal = unpackNormal3(normal.xy);

    data = texture(deferred_data,tc).xyz;


//    normal = normal*2.0f - 1.0f;
}


