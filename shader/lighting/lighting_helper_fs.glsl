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

const vec2 poissonDisk[4] = vec2[](
   vec2( -0.94201624, -0.39906216 ),
   vec2( 0.94558609, -0.76890725 ),
   vec2( -0.094184101, -0.92938870 ),
   vec2( 0.34495938, 0.29387760 )
 );

const vec2 poissonDisk64[64] = vec2[](
    vec2(-0.613392, 0.617481),
    vec2(0.170019, -0.040254),
    vec2(-0.299417, 0.791925),
    vec2(0.645680, 0.493210),
    vec2(-0.651784, 0.717887),
    vec2(0.421003, 0.027070),
    vec2(-0.817194, -0.271096),
    vec2(-0.705374, -0.668203),
    vec2(0.977050, -0.108615),
    vec2(0.063326, 0.142369),
    vec2(0.203528, 0.214331),
    vec2(-0.667531, 0.326090),
    vec2(-0.098422, -0.295755),
    vec2(-0.885922, 0.215369),
    vec2(0.566637, 0.605213),
    vec2(0.039766, -0.396100),
    vec2(0.751946, 0.453352),
    vec2(0.078707, -0.715323),
    vec2(-0.075838, -0.529344),
    vec2(0.724479, -0.580798),
    vec2(0.222999, -0.215125),
    vec2(-0.467574, -0.405438),
    vec2(-0.248268, -0.814753),
    vec2(0.354411, -0.887570),
    vec2(0.175817, 0.382366),
    vec2(0.487472, -0.063082),
    vec2(-0.084078, 0.898312),
    vec2(0.488876, -0.783441),
    vec2(0.470016, 0.217933),
    vec2(-0.696890, -0.549791),
    vec2(-0.149693, 0.605762),
    vec2(0.034211, 0.979980),
    vec2(0.503098, -0.308878),
    vec2(-0.016205, -0.872921),
    vec2(0.385784, -0.393902),
    vec2(-0.146886, -0.859249),
    vec2(0.643361, 0.164098),
    vec2(0.634388, -0.049471),
    vec2(-0.688894, 0.007843),
    vec2(0.464034, -0.188818),
    vec2(-0.440840, 0.137486),
    vec2(0.364483, 0.511704),
    vec2(0.034028, 0.325968),
    vec2(0.099094, -0.308023),
    vec2(0.693960, -0.366253),
    vec2(0.678884, -0.204688),
    vec2(0.001801, 0.780328),
    vec2(0.145177, -0.898984),
    vec2(0.062655, -0.611866),
    vec2(0.315226, -0.604297),
    vec2(-0.780145, 0.486251),
    vec2(-0.371868, 0.882138),
    vec2(0.200476, 0.494430),
    vec2(-0.494552, -0.711051),
    vec2(0.612476, 0.705252),
    vec2(-0.578845, -0.768792),
    vec2(-0.772454, -0.090976),
    vec2(0.504440, 0.372295),
    vec2(0.155736, 0.065157),
    vec2(0.391522, 0.849605),
    vec2(-0.620106, -0.328104),
    vec2(0.789239, -0.419965),
    vec2(-0.545396, 0.538133),
    vec2(-0.178564, -0.596057)
);




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

float calculateShadowPCF(sampler2DShadow tex, vec3 position, float outside){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;

     float visibility = 1.0f;
     if ((shadowPos.x < 0 || shadowPos.x > 1 || shadowPos.y < 0 || shadowPos.y > 1 || shadowPos.z < 0 || shadowPos.z > 1)){
         visibility = outside;
     }else{
        visibility = 0.0f;
         float poissonRadius = 1.5f;
         float pixelOffsetRadius = 0.5f;
         float pcfSize = 0.5f;
         float samples = (pcfSize*2.0+1.0f) * (pcfSize*2.0f+1.0f);

        int i = 0;
         for(float u = -pcfSize ; u <= pcfSize ; u = u+1.0){
             for(float v = -pcfSize ; v <= pcfSize ; v = v+1.0){
                 int index = int(64.0*random(vec4(gl_FragCoord.xyy, i)))%64;

                 vec2 offset = (poissonDisk64[index]*poissonRadius+vec2(u,v)*pixelOffsetRadius)*shadowMapSize.zw*1.0f;
                 visibility += texture(tex, shadowPos.xyz + vec3(offset,0));
                 i++;
             }
         }
         visibility *= 1.0/(samples);
     }
     return visibility;

}

vec2 roundToShadowMapPixel(vec2 uv, vec4 shadowMapSize){
    return (round(uv * shadowMapSize.xy + 0.5f)-0.5f) *  shadowMapSize.zw;
}

vec2 roundToShadowMapPixelCorner(vec2 uv, vec4 shadowMapSize){
    return round(uv * shadowMapSize.xy) *  shadowMapSize.zw;
}

float calculateShadowPCF2(sampler2DShadow tex, vec3 position, float outside){
    vec4 shadowPos = depthBiasMV * vec4(position,1);
    shadowPos = shadowPos/shadowPos.w;

    shadowPos.xy = roundToShadowMapPixelCorner(shadowPos.xy,shadowMapSize);

     float visibility = 1.0f;
     if ((shadowPos.x < 0 || shadowPos.x > 1 || shadowPos.y < 0 || shadowPos.y > 1 || shadowPos.z < 0 || shadowPos.z > 1)){
         visibility = outside;
     }else{
        visibility = 0.0f;
         float pcfSize = 1.0f;
         float samples = (pcfSize*2.0+1.0f) * (pcfSize*2.0f+1.0f);


         for(float u = -pcfSize ; u <= pcfSize ; u = u+1.0){
             for(float v = -pcfSize ; v <= pcfSize ; v = v+1.0){
                 vec2 offset = vec2(u,v)*(shadowMapSize.zw*2.0f);
                 visibility += texture(tex, shadowPos.xyz + vec3(offset,0));
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
