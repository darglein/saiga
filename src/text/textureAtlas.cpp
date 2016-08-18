#include "saiga/text/textureAtlas.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/text/fontLoader.h"
#include <algorithm>
#include <fstream>
#include <FreeImagePlus.h>
#include "saiga/util/assert.h"


#define NOMINMAX
#undef max
#undef min



TextureAtlas::TextureAtlas(){

}

TextureAtlas::~TextureAtlas()
{
    delete textureAtlas;
}

void TextureAtlas::loadFont(const std::string &font, int fontSize, int quality, int searchRange, bool bufferToFile, const std::vector<Unicode::UnicodeBlock> &blocks){
    std::string blockString;
    for(Unicode::UnicodeBlock ub : blocks){
        blockString = blockString + std::to_string(ub.start);
    }


    uniqueFontString = font+"."+std::to_string(fontSize)+"_"+std::to_string(quality)+"_"+std::to_string(searchRange)+"_"+blockString+".sdf";

    //add an 'empty' character for new line
    characterInfoMap['\n'] = character_info();

    if(bufferToFile && readAtlasFromFiles()){
//        cout << "maxCharacter: " << maxCharacter << endl;
        return;
    }

    cout<<"Generating new SDF Text Atlas: "<<uniqueFontString<<endl;
    quality = quality*2+1;

    FontLoader fl(font,blocks);
    fl.loadMonochromatic(fontSize*quality,(1+quality)*searchRange);
    //    fl.writeGlyphsToFiles("debug/fonts/");

    Image img;
    createTextureAtlas(img,fl.glyphs,quality,quality*searchRange);
    //    fl.writeGlyphsToFiles("debug/fonts2/");

    if(bufferToFile){
        writeAtlasToFiles(img);
    }

//    cout << "maxCharacter: " << maxCharacter << endl;




    textureAtlas = new Texture();
    textureAtlas->fromImage(img);
    textureAtlas->generateMipmaps();
}

const TextureAtlas::character_info &TextureAtlas::getCharacterInfo(int c){
    auto it = characterInfoMap.find(c);
    if(it==characterInfoMap.end()){
        cerr<<"TextureAtlas::getCharacterInfo: Invalid character '"<<std::hex<<c<<"'"<<endl;
        return invalidCharacter;
    }
    return (*it).second;
}



void TextureAtlas::createTextureAtlas(Image &outImg, std::vector<FontLoader::Glyph> &glyphs, int downsample, int searchRadius)
{
    numCharacters = glyphs.size();
    cout<<"TextureAtlas::createTextureAtlas: Number of glyphs = "<<numCharacters<<endl;
    padGlyphsToDivisor(glyphs,downsample);
    convertToSDF(glyphs,downsample,searchRadius);
    calculateTextureAtlasLayout(glyphs);

    outImg.Format() = ImageFormat(1,8);
//    outImg.bitDepth = 8;
//    outImg.channels = 1;
    outImg.width = atlasWidth;
    outImg.height = atlasHeight;
    outImg.create();
    outImg.makeZero();

    cout<<"AtlasWidth "<<atlasWidth<<" AtlasHeight "<<atlasHeight<<endl;

    for(FontLoader::Glyph &g : glyphs) {
        character_info &info = characterInfoMap[g.character];
        outImg.setSubImage(info.atlasPos.x,info.atlasPos.y,*g.bitmap);
    }
    numCharacters = characterInfoMap.size();
}

void TextureAtlas::calculateTextureAtlasLayout(std::vector<FontLoader::Glyph> &glyphs)
{
    int charsPerRow = glm::ceil(glm::sqrt((float)glyphs.size()));

    atlasHeight = charPaddingY;
    atlasWidth = 0;

    maxCharacter.makeNegative();

    for(int cy = 0 ; cy < charsPerRow ; ++cy){
        int currentW = charPaddingX;
        int currentH = charPaddingY;

        for(int cx = 0 ; cx < charsPerRow ; ++cx){
            int i = cy * charsPerRow + cx;
            if(i>=(int)glyphs.size())
                break;

            FontLoader::Glyph &g = glyphs[i];

            character_info info;

            info.character = g.character;
            info.advance.x = g.advance.x;
            info.advance.y = g.advance.y;

            info.size.x = g.size.x;
            info.size.y = g.size.y;

            info.offset.x = g.offset.x;
            info.offset.y = g.offset.y - info.size.y; //freetype uses an y inverted glyph coordinate system

            maxCharacter.min = glm::min(maxCharacter.min,vec3(info.offset.x,info.offset.y,0));
            maxCharacter.max = glm::max(maxCharacter.max,vec3(info.offset.x+info.size.x,info.offset.y+info.size.y,0));



            info.atlasPos.x = currentW;
            info.atlasPos.y = atlasHeight;

            characterInfoMap[g.character] = info;

            currentW += g.bitmap->width+charPaddingX;
            currentH = std::max(currentH, (int)g.bitmap->height);


        }
        atlasWidth = std::max(currentW, atlasWidth);
        atlasHeight += currentH+charPaddingY;
    }

    //calculate the texture coordinates
    for(FontLoader::Glyph &g : glyphs) {
        character_info &info = characterInfoMap[g.character];
        float tx = (float)info.atlasPos.x / (float)atlasWidth;
        float ty = (float)info.atlasPos.y / (float)atlasHeight;

        info.tcMin = vec2(tx,ty);
        info.tcMax = vec2(tx+(float)info.size.x/(float)atlasWidth,ty+(float)info.size.y/(float)atlasHeight);
    }
}

void TextureAtlas::padGlyphsToDivisor(std::vector<FontLoader::Glyph> &glyphs, int divisor)
{
    for(FontLoader::Glyph &g : glyphs) {
        int w = g.bitmap->width;
        int h = g.bitmap->height;

        //        cout<<"old "<<w<<","<<h;
        w += (divisor - (w % divisor)) % divisor;
        h += (divisor - (h % divisor)) % divisor;
        //        cout<<" new "<<w<<","<<h<<endl;
        g.bitmap->resizeCopy(w,h);




    }
}

void TextureAtlas::convertToSDF(std::vector<FontLoader::Glyph> &glyphs, int divisor, int searchRadius)
{
    assert(divisor%2==1);
    int halfDivisor = (divisor / 2);

    std::vector<glm::ivec2> samplePositions = generateSDFsamples(searchRadius);
    for(FontLoader::Glyph &g : glyphs) {
        Image* sdfGlyph = new Image();
        sdfGlyph->width = g.bitmap->width / divisor;
        sdfGlyph->height = g.bitmap->height / divisor;
//        sdfGlyph->channels = g.bitmap->channels;
//        sdfGlyph->bitDepth = g.bitmap->bitDepth;
        sdfGlyph->Format() = g.bitmap->Format();
        sdfGlyph->create();
        sdfGlyph->makeZero();

        g.advance /= divisor;
        g.offset /= divisor;
        g.size /= divisor;


        for(int y = 0 ; y < (int)sdfGlyph->height ; ++y){
            for(int x = 0 ; x < (int)sdfGlyph->width ; ++x){

                //center of sdftexel in original image
                int bx = x * divisor + halfDivisor;
                int by = y * divisor + halfDivisor;
                float d = 12345;
                unsigned char current = g.bitmap->getPixel<unsigned char>(bx,by);
                for(glm::ivec2 s : samplePositions){
                    glm::ivec2 ps = glm::ivec2(bx,by) + s;
                    ps = glm::clamp(ps,glm::ivec2(0),glm::ivec2(g.bitmap->width-1,g.bitmap->height-1));
                    unsigned char other = g.bitmap->getPixel<unsigned char>(ps.x,ps.y);
                    if(current!=other){
                        d = glm::sqrt((float)(s.x*s.x+s.y*s.y));
                        break;
                    }
                }

                //map to 0-1
                d = d / (float)searchRadius;
                d = glm::clamp(d,0.0f,1.0f);
                d = d * 0.5f;

                //set 0.5 to border and >0.5 to inside and <0.5 to outside
                if(current){
                    d = d + 0.5f;
                }else{
                    d = 0.5f - d;
                }

                unsigned char out = d * 255.0f;
                sdfGlyph->setPixel(x,y,out);
            }
        }

        delete g.bitmap;
        g.bitmap = sdfGlyph;
    }
}

std::vector<glm::ivec2> TextureAtlas::generateSDFsamples(int searchRadius)
{

    std::vector<glm::ivec2> samplePositions;
    for(int x = -searchRadius ; x <= searchRadius ; ++x){
        for(int y = -searchRadius ; y <= searchRadius ; ++y){
            if(x!=0 || y!=0)
                samplePositions.emplace_back(x,y);
        }
    }
    std::sort(samplePositions.begin(),samplePositions.end(),[](const glm::ivec2 &a,const glm::ivec2 &b)->bool
    {
        return (a.x*a.x+a.y*a.y)<(b.x*b.x+b.y*b.y);
    });

    //remove samples further away then searchRadius
    int samples = 0;
    for(;samples<(int)samplePositions.size();samples++){
        glm::ivec2 a = samplePositions[samples];
        if(glm::sqrt((float)(a.x*a.x+a.y*a.y))>searchRadius){
            break;
        }
    }
    samplePositions.resize(samples);

    return samplePositions;
}

void TextureAtlas::writeAtlasToFiles(Image& img)
{
    std::string str = uniqueFontString + ".png";
    if(!TextureLoader::instance()->saveImage(str,img)){
        cout<<"could not save "<<str<<endl;
    }

    std::ofstream stream (uniqueFontString,std::ofstream::binary);
    stream.write((char*)&numCharacters,sizeof(uint32_t));
    int i = 0;
    for(std::pair<const int,character_info> ci : characterInfoMap){
        stream.write((char*)&ci.second,sizeof(character_info));
        i ++;
    }
//    cout << i << " Characters written to file." << endl;
    assert(i == numCharacters);

    stream.write((char*)&maxCharacter,sizeof(aabb));
    stream.close();

}

bool TextureAtlas::readAtlasFromFiles()
{
    std::ifstream stream (uniqueFontString,std::ifstream::binary);
    if(!stream.is_open())
        return false;
    stream.read((char*)&numCharacters,sizeof(uint32_t));
    for(int i = 0 ; i < numCharacters ; ++i){
        character_info ci;
        stream.read((char*)&ci,sizeof(character_info));
        characterInfoMap[ci.character] = ci;
    }
//    cout << numCharacters << " Characters read from file." << endl;
    stream.read((char*)&maxCharacter,sizeof(aabb));
    stream.close();


    std::string str = uniqueFontString + ".png";
    Image img;
    if(!TextureLoader::instance()->loadImage(str,img)){
        return false;
    }

    textureAtlas = new Texture();
    textureAtlas->fromImage(img);
    textureAtlas->generateMipmaps();

    return true;

}
