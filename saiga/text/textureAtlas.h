#pragma once


#include "saiga/config.h"
#include "saiga/util/glm.h"
#include "saiga/geometry/aabb.h"
#include "saiga/text/fontLoader.h"
#include <iostream>

class basic_Texture_2D;


class SAIGA_GLOBAL TextureAtlas{
public:
    struct character_info {
        glm::vec2 advance; //distance to the origin of the next character
        glm::vec2 offset;  //offset of the bitmap position to the origin of this character
        glm::vec2 size; //size of bitmap

        glm::ivec2 atlasPos = glm::ivec2(0); //position of this character in the texture atlas
//        int atlasX = 0, atlasY = 0;

        vec2 tcMin,tcMax;
    } ;

    TextureAtlas();
    ~TextureAtlas();

    /**
     * Loads a True Type font (.ttf) with libfreetype.
     * This will create the textureAtlas, so it has to be called before any ussage.
     */
    void loadFont(const std::string &font, int fontSize=40, int quality=4, int searchRange=5, bool bufferToFile=false);

    /**
     * Returns the bounding box that could contain every character in this font.
     */
    aabb getMaxCharacter(){return maxCharacter;}

    /**
     * Returns the actual opengl texture.
     */
    basic_Texture_2D *getTexture(){return textureAtlas;}

    /**
     * Returns information to a specific character in this font.
     */
    const character_info& getCharacterInfo(int c){ return characters[c];}

    /**
     * Returns the distance between the bottom of one line to the bottom of the next line.
     */
    float getLineSpacing(){ return (maxCharacter.max - maxCharacter.min).y + additionalLineSpacing;}


    //these values are added to the default line and character spacings.
    //a positive value moves the characters and lines further apart
    //and a negative values pulls them closer together.
    float additionalLineSpacing = 0;
    float additionalCharacterSpacing = 0;
private:

    //distance between characters in texture atlas
    int charPaddingX = 0;
    int charPaddingY = 0;

    int atlasHeight;
    int atlasWidth;

    static const int maxNumCharacters = 256;
    character_info characters[maxNumCharacters];

    basic_Texture_2D *textureAtlas = nullptr;
    aabb maxCharacter;
    std::string font;
    std::string uniqueFontString;


    void createTextureAtlas(Image &outImg, std::vector<FontLoader::Glyph> &glyphs, int downsample, int searchRadius);
    void calculateTextureAtlasLayout(std::vector<FontLoader::Glyph> &glyphs);
    void padGlyphsToDivisor(std::vector<FontLoader::Glyph> &glyphs, int divisor);
    void convertToSDF(std::vector<FontLoader::Glyph> &glyphs, int divisor, int searchRadius);
    std::vector<glm::ivec2> generateSDFsamples(int searchRadius);

    void writeAtlasToFiles(Image &img);
    bool readAtlasFromFiles();

};
