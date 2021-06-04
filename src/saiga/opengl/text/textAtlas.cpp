/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/text/textAtlas.h"

#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/text/fontLoader.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/texture/TextureLoader.h"

#include <algorithm>
#include <fstream>

namespace Saiga
{
#define NOMINMAX
#undef max
#undef min



TextAtlas::TextAtlas() {}

TextAtlas::~TextAtlas()
{
    //    delete textureAtlas;
}

void TextAtlas::loadFont(const std::string& _font, int fontSize, int quality, int searchRange, bool bufferToFile,
                         const std::vector<Unicode::UnicodeBlock>& blocks)
{
    std::string font = SearchPathes::font(_font);
    if (font == "")
    {
        std::cerr << "Could not open file " << _font << std::endl;
        std::cerr << SearchPathes::font << std::endl;
        SAIGA_ASSERT(0);
    }

    std::string blockString;
    for (Unicode::UnicodeBlock ub : blocks)
    {
        blockString = blockString + std::to_string(ub.start);
    }


    uniqueFontString = font + "." + std::to_string(fontSize) + "_" + std::to_string(quality) + "_" +
                       std::to_string(searchRange) + "_" + blockString + ".sdf";

    // add an 'empty' character for new line
    characterInfoMap['\n'] = character_info();

    if (bufferToFile && readAtlasFromFiles())
    {
        initFont();
        return;
    }

    std::cout << "Generating new SDF Text Atlas: " << uniqueFontString << std::endl;
    quality = quality * 2 + 1;

    FontLoader fl(font, blocks);
    fl.loadMonochromatic(fontSize * quality, (1 + quality) * searchRange);
    //        fl.writeGlyphsToFiles("debug/fonts/");


    createTextureAtlas(fl.glyphs, quality, quality * searchRange);
    //        fl.writeGlyphsToFiles("debug/fonts2/");



    if (bufferToFile)
    {
        writeAtlasToFiles();
    }

    //    std::cout << "maxCharacter: " << maxCharacter << std::endl;



    textureAtlas = std::make_shared<Texture>(atlas);
    textureAtlas->generateMipmaps();

    initFont();
}


void TextAtlas::initFont()
{
    invalidCharacter = getCharacterInfo('?');
}


const TextAtlas::character_info& TextAtlas::getCharacterInfo(int c)
{
    auto it = characterInfoMap.find(c);
    if (it == characterInfoMap.end())
    {
        std::cerr << "TextureAtlas::getCharacterInfo: Invalid character '" << std::hex << c << "'" << std::endl;
        return invalidCharacter;
    }
    return (*it).second;
}



void TextAtlas::createTextureAtlas(std::vector<FontLoader::Glyph>& glyphs, int downsample, int searchRadius)
{
    numCharacters = glyphs.size();
    std::cout << "TextureAtlas::createTextureAtlas: Number of glyphs = " << numCharacters << std::endl;
    padGlyphsToDivisor(glyphs, downsample);



    convertToSDF(glyphs, downsample, searchRadius);



    calculateTextureAtlasLayout(glyphs);

    atlas.create(atlasHeight, atlasWidth);
    atlas.makeZero();

    std::cout << "AtlasWidth " << atlasWidth << " AtlasHeight " << atlasHeight << std::endl;

    for (FontLoader::Glyph& g : glyphs)
    {
        character_info& info = characterInfoMap[g.character];
        atlas.getImageView().setSubImage(info.atlasPos[1], info.atlasPos[0], g.bitmap.getImageView());
    }
    numCharacters = characterInfoMap.size();
}

void TextAtlas::calculateTextureAtlasLayout(std::vector<FontLoader::Glyph>& glyphs)
{
    int charsPerRow = ceil(sqrt((float)glyphs.size()));

    atlasHeight = charPaddingY;
    atlasWidth  = 0;

    maxCharacter.makeNegative();

    for (int cy = 0; cy < charsPerRow; ++cy)
    {
        int currentW = charPaddingX;
        int currentH = charPaddingY;

        for (int cx = 0; cx < charsPerRow; ++cx)
        {
            int i = cy * charsPerRow + cx;
            if (i >= (int)glyphs.size()) break;

            FontLoader::Glyph& g = glyphs[i];

            character_info info;

            info.character  = g.character;
            info.advance[0] = g.advance[0];
            info.advance[1] = g.advance[1];

            info.size[0] = g.size[0];
            info.size[1] = g.size[1];

            info.offset[0] = g.offset[0];
            info.offset[1] = g.offset[1] - info.size[1];  // freetype uses an y inverted glyph coordinate system

            //            maxCharacter.min = std::min(maxCharacter.min, vec3(info.offset[0], info.offset[1], 0));
            //            maxCharacter.max =
            //                std::max(maxCharacter.max, vec3(info.offset[0] + info.size[0], info.offset[1] +
            //                info.size[1], 0));

            maxCharacter.min = maxCharacter.min.array().min(vec3(info.offset[0], info.offset[1], 0).array());
            maxCharacter.max = maxCharacter.max.array().min(
                vec3(info.offset[0] + info.size[0], info.offset[1] + info.size[1], 0).array());



            info.atlasPos[0] = currentW;
            info.atlasPos[1] = atlasHeight;

            characterInfoMap[g.character] = info;

            currentW += g.bitmap.width + charPaddingX;
            currentH = std::max(currentH, (int)g.bitmap.height);
        }
        atlasWidth = std::max(currentW, atlasWidth);
        atlasHeight += currentH + charPaddingY;
    }

    // calculate the texture coordinates
    for (FontLoader::Glyph& g : glyphs)
    {
        character_info& info = characterInfoMap[g.character];
        float tx             = (float)info.atlasPos[0] / (float)atlasWidth;
        float ty             = (float)info.atlasPos[1] / (float)atlasHeight;

        info.tcMin = vec2(tx, ty);
        info.tcMax = vec2(tx + (float)info.size[0] / (float)atlasWidth, ty + (float)info.size[1] / (float)atlasHeight);
    }
}

void TextAtlas::padGlyphsToDivisor(std::vector<FontLoader::Glyph>& glyphs, int divisor)
{
    for (FontLoader::Glyph& g : glyphs)
    {
        int w = g.bitmap.width;
        int h = g.bitmap.height;

        //                std::cout<<"old "<<w<<","<<h;
        w += (divisor - (w % divisor)) % divisor;
        h += (divisor - (h % divisor)) % divisor;

        SAIGA_ASSERT(w % divisor == 0);
        SAIGA_ASSERT(h % divisor == 0);

        //                std::cout<<" new "<<w<<","<<h<<endl;

        TemplatedImage<unsigned char> paddedImage(h, w);
        paddedImage.makeZero();

        auto subImg = paddedImage.getImageView().subImageView(0, 0, g.bitmap.height, g.bitmap.width);
        g.bitmap.getImageView().copyTo(subImg);

        g.bitmap = paddedImage;

        //                 g.bitmap.save("debug/font/"+to_string(g.character)+"_3.png");
        //        g.bitmap->resizeCopy(w,h);
        //        SAIGA_ASSERT(0);
    }
}

void TextAtlas::convertToSDF(std::vector<FontLoader::Glyph>& glyphs, int divisor, int searchRadius)
{
    SAIGA_ASSERT(divisor % 2 == 1);
    int halfDivisor = (divisor / 2);

    std::vector<ivec2> samplePositions = generateSDFsamples(searchRadius);
    for (FontLoader::Glyph& g : glyphs)
    {
        //        Image* sdfGlyph = new Image();
        TemplatedImage<unsigned char> sdfGlyph;
        sdfGlyph.width  = g.bitmap.width / divisor;
        sdfGlyph.height = g.bitmap.height / divisor;
        //        sdfGlyph->channels = g.bitmap->channels;
        //        sdfGlyph->bitDepth = g.bitmap->bitDepth;
        //        sdfGlyph->Format() = g.bitmap->Format();
        //        SAIGA_ASSERT(0);
        sdfGlyph.create();
        sdfGlyph.makeZero();

        g.advance /= divisor;
        g.offset /= divisor;
        g.size /= divisor;


        for (int y = 0; y < (int)sdfGlyph.height; ++y)
        {
            for (int x = 0; x < (int)sdfGlyph.width; ++x)
            {
                // center of sdftexel in original image
                int bx  = x * divisor + halfDivisor;
                int by  = y * divisor + halfDivisor;
                float d = 12345;

                bool current = g.bitmap(by, bx);
                for (ivec2 s : samplePositions)
                {
                    ivec2 ps   = ivec2(bx, by) + s;
                    ps         = clamp(ps, ivec2(0, 0), ivec2(g.bitmap.width - 1, g.bitmap.height - 1));
                    bool other = g.bitmap(ps[1], ps[0]);
                    if (current != other)
                    {
                        d = sqrt((float)(s[0] * s[0] + s[1] * s[1]));
                        break;
                    }
                }

                // SAIGA_ASSERT(d < 10000);


                // map to 0-1
                d = d / (float)searchRadius;
                d = clamp(d, 0.0f, 1.0f);
                d = d * 0.5f;

                // set 0.5 to border and >0.5 to inside and <0.5 to outside
                if (current)
                {
                    d = d + 0.5f;
                }
                else
                {
                    d = 0.5f - d;
                }

                //                std::cout << "(" << x << ", " << y << ") = " << d << std::endl;

                unsigned char out = d * 255.0f;
                sdfGlyph(y, x)    = out;
            }
        }

        //        g.bitmap.save(to_string(g.character) + "_4.png");
        //        sdfGlyph.save(to_string(g.character) + "_5.png");

        //        exit(0);
        //        delete g.bitmap;
        g.bitmap = sdfGlyph;
    }
}

std::vector<ivec2> TextAtlas::generateSDFsamples(int searchRadius)
{
    std::vector<ivec2> samplePositions;
    for (int x = -searchRadius; x <= searchRadius; ++x)
    {
        for (int y = -searchRadius; y <= searchRadius; ++y)
        {
            if (x != 0 || y != 0) samplePositions.emplace_back(x, y);
        }
    }
    std::sort(samplePositions.begin(), samplePositions.end(), [](const ivec2& a, const ivec2& b) -> bool {
        return (a[0] * a[0] + a[1] * a[1]) < (b[0] * b[0] + b[1] * b[1]);
    });

    // remove samples further away then searchRadius
    int samples = 0;
    for (; samples < (int)samplePositions.size(); samples++)
    {
        ivec2 a = samplePositions[samples];
        if (sqrt((float)(a[0] * a[0] + a[1] * a[1])) > searchRadius)
        {
            break;
        }
    }
    samplePositions.resize(samples);

    return samplePositions;
}

void TextAtlas::writeAtlasToFiles()
{
    std::string str = uniqueFontString + ".png";


    atlas.save(str);


    std::ofstream stream(uniqueFontString, std::ofstream::binary);
    stream.write((char*)&numCharacters, sizeof(uint32_t));
    int i = 0;
    for (std::pair<const int, character_info> ci : characterInfoMap)
    {
        stream.write((char*)&ci.second, sizeof(character_info));
        i++;
    }
    //    std::cout << i << " Characters written to file." << std::endl;
    SAIGA_ASSERT(i == numCharacters);

    stream.write((char*)&maxCharacter, sizeof(AABB));
    stream.close();
}

bool TextAtlas::readAtlasFromFiles()
{
    std::ifstream stream(uniqueFontString, std::ifstream::binary);
    if (!stream.is_open()) return false;
    stream.read((char*)&numCharacters, sizeof(uint32_t));
    for (int i = 0; i < numCharacters; ++i)
    {
        character_info ci;
        stream.read((char*)&ci, sizeof(character_info));
        characterInfoMap[ci.character] = ci;
    }
    //    std::cout << numCharacters << " Characters read from file." << std::endl;
    stream.read((char*)&maxCharacter, sizeof(AABB));
    stream.close();


    std::string str = uniqueFontString + ".png";


    atlas.load(str);


    //    TextureLoader::instance()->saveImage("asdf2.png",img);

    textureAtlas = std::make_shared<Texture>(atlas, false, false);
    textureAtlas->generateMipmaps();

    std::cout << "readAtlasFromFiles: " << uniqueFontString << " numCharacters: " << numCharacters << std::endl;

    return true;
}

}  // namespace Saiga
