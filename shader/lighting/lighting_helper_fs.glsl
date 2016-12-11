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

//uniform sampler2DMS deferred_diffuse;
//uniform sampler2DMS deferred_normal;
//uniform sampler2DMS deferred_depth;
//uniform sampler2DMS deferred_position;
//uniform sampler2DMS deferred_data;

uniform vec2 screen_size;

uniform vec4 lightColorDiffuse; //rgba, rgb=color, a=intensity [0,1]
uniform vec4 lightColorSpecular; //rgba, rgb=color, a=intensity [0,1]

uniform vec4 shadowMapSize;  //vec4(w,h,1/w,1/h)


float random(vec4 seed4){
    float dot_product = dot(seed4, vec4(12.9898,78.233,45.164,94.673));
    return fract(sin(dot_product) * 43758.5453);
}

vec2 CalcTexCoord()
{
   return gl_FragCoord.xy / screen_size;
}



float intensityDiffuse(vec3 normal, vec3 lightDir){
    return max(dot(normal,lightDir), 0.0);
}

float intensitySpecular(vec3 position, vec3 normal, vec3 lightDir, float exponent){
    vec3 viewDir = normalize(-position);

#if 0
    // phong shading
//    vec3 reflectDir = normalize(-reflect(lightDir,normal));
    vec3 reflectDir = -reflect(lightDir,normal);
    float specAngle = max(dot(reflectDir,viewDir),0.0);
    float i = pow(specAngle, exponent);
#else
    // blinn phong shading
    vec3 halfDir = normalize(lightDir + viewDir);
    float specAngle = max(dot(halfDir, normal), 0.0);
    float i = pow(specAngle, exponent*4);
#endif

    return clamp(i,0.0f,1.0f);
}



float getAttenuation(vec3 attenuation, float distance, float radius){
//    if(distance > 5.6f)
//    return 1.0f;
//    return 0.0f;
    //normalize the distance, so the attenuation is independent of the radius
    float x = distance / radius;
    //make sure that we return 0 if distance > radius, otherwise we would get an hard edge
    float smoothBorder = smoothstep(1.0f,0.9f,x);
    return smoothBorder / (attenuation.x +
                    attenuation.y * x +
                    attenuation.z * x * x);
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

float linearizeDepth(in float depth, in mat4 projMatrix) {
        return projMatrix[3][2] / (depth - projMatrix[2][2]);
}

float linearDepth(float depth, float farplane, float nearplane){
//    float f=60.0f;
//    float n = 1.0f;
    return(2 * nearplane) / (farplane + nearplane - depth * (farplane - nearplane));
}


vec3 reconstructPosition(float d, vec2 tc){
    vec4 p = vec4(tc.x,tc.y,d,1)*2.0f - 1.0f;
    p = invProj * p;
    return p.xyz/p.w;
}



void getGbufferData(out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data, int sampleId){
    vec2 tc = CalcTexCoord();
    ivec2 tci = ivec2(gl_FragCoord.xy);

    color = texelFetch( deferred_diffuse, tci ,sampleId).rgb;

    depth = texelFetch( deferred_depth, tci ,sampleId).r;
    position = reconstructPosition(depth,tc);

    normal = texelFetch( deferred_normal, tci,sampleId).xyz;
    normal = unpackNormal3(normal.xy);

    data = texelFetch(deferred_data,tci,sampleId).xyz;
}


void getGbufferData(vec2 tc, out vec3 color,out  vec3 position, out float depth, out vec3 normal, out vec3 data){

    color = texture( deferred_diffuse, tc).rgb;

    depth = texture( deferred_depth, tc).r;
    position = reconstructPosition(depth,tc);

    normal = texture( deferred_normal, tc).xyz;
    normal = unpackNormal3(normal.xy);

    data = texture(deferred_data,tc).xyz;
}

#include "shadows.glsl"
