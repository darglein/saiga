/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/translator.h"

#include "saiga/core/util/assert.h"

#include "internal/noGraphicsAPI.h"

#include <algorithm>
#include <fstream>
#include <iostream>

namespace Saiga
{
Translator translator;


std::vector<std::string> split(const std::string& s, char delim);


std::string Translator::replace(std::string str, const std::string& search, const std::string& replace)
{
    size_t pos = 0;
    while ((pos = str.find(search, pos)) != std::string::npos)
    {
        str.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return str;
}

std::string Translator::escapeSpecialCharacters(std::string str)
{
    std::string esc(1, escapesymbol);
    str = replace(str, std::string(1, escapesymbol), esc + escapesymbol);
    str = replace(str, std::string(1, spacer), esc + spacer);
    str = replace(str, std::string(1, '\n'), esc + newlinesymbol);
    return str;
}

std::string Translator::unescapeSpecialCharacters(std::string str)
{
    std::string esc(1, escapesymbol);

    size_t pos = 0;
    while ((pos = str.find(esc, pos)) != std::string::npos)
    {
        str.erase(str.begin() + pos);
        char nextChar    = str[pos];
        char replacement = nextChar;
        if (nextChar == 'n')
        {
            replacement = '\n';
        }
        str[pos] = replacement;
        pos++;
    }
    return str;
}

void Translator::addTranslation(const Translator::TranslationEntry& te)
{
    if (translationMap.find(te.key) != translationMap.end())
    {
        std::cerr << "Key " << te.key << " is already in the translation map." << std::endl;
        SAIGA_ASSERT(0);
    }
    translationMap[te.key] = te;
}

std::vector<Translator::TranslationEntry> Translator::getTranslationVector()
{
    std::vector<TranslationEntry> erg;
    for (auto iterator = translationMap.begin(); iterator != translationMap.end(); iterator++)
    {
        TranslationEntry te = iterator->second;
        erg.push_back(te);
    }
    std::sort(erg.begin(), erg.end(), [](TranslationEntry a, TranslationEntry b) -> bool { return a.key < b.key; });
    return erg;
}


bool Translator::readTranslationFile()
{
    std::cout << "Reading translation file " << translationFile << std::endl;

    std::fstream stream;

    try
    {
        stream.open(translationFile, std::fstream::in);
        if (!stream.good())
        {
            std::cout << "Warning Translator: file does not exist! " + translationFile << std::endl;
            return false;
        }
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return false;
    }

    int lineNumber = 0;
    try
    {
        for (std::string line; std::getline(stream, line); lineNumber++)
        {
            TranslationEntry te;

            if (stream.eof())
            {
                break;
            }

            // remove carriage return from windows
            line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());

            utf32string utf32line = Encoding::UTF8toUTF32(line);

            if ((int)utf32line.size() == 0 || (int)utf32line.front() == comment) continue;

            // remove new entry symbol
            if ((int)utf32line.front() == newEntry)
            {
                utf32line.erase(utf32line.begin());
                te.newEntry = true;
            }

            utf32line.erase(std::remove(utf32line.begin(), utf32line.end(), '\r'), utf32line.end());


            // find positions of spacers (without leading escape symbols)
            std::vector<int> spacerPos;
            int pos       = 0;
            uint32_t prev = 0;
            for (auto c : utf32line)
            {
                if ((int)c == spacer && (int)prev != escapesymbol)
                {
                    spacerPos.push_back(pos);
                }
                prev = c;
                pos++;
            }

//            //if no second spacer is found just add one to the end of the line
//            if (spacerPos.size() == 1){
//                spacerPos.push_back(utf32line.size());
//                utf32line.push_back(spacer);
//            }
#if defined(SAIGA_DEBUG)
            if (spacerPos.size() != 2)
            {
                std::cout << "Line " << lineNumber << ": " << Encoding::UTF32toUTF8(utf32line) << std::endl;
            }
#endif
            SAIGA_ASSERT(spacerPos.size() == 2);


            utf32string key(utf32line.begin(), utf32line.begin() + spacerPos[0]);
            utf32string trans(utf32line.begin() + spacerPos[0] + 1, utf32line.begin() + spacerPos[1]);
            utf32string note(utf32line.begin() + spacerPos[1] + 1, utf32line.end());

            SAIGA_ASSERT(key.size() > 0);

            te.key         = unescapeSpecialCharacters(Encoding::UTF32toUTF8(key));
            te.translation = unescapeSpecialCharacters(Encoding::UTF32toUTF8(trans));
            te.note        = unescapeSpecialCharacters(Encoding::UTF32toUTF8(note));
            //            std::cout << lineNumber << " " << te.key << spacer << te.translation << spacer << te.note <<
            //            std::endl;
            addTranslation(te);
        }
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        return false;
    }
    return true;
}


void Translator::init(const std::string& file)
{
    translationFile = file;
    translationMap.clear();

    readTranslationFile();
}

void Translator::writeToFile()
{
    std::string outFile                   = translationFile;
    std::vector<TranslationEntry> entries = getTranslationVector();
    std::fstream stream;

    try
    {
        stream.open(outFile, std::fstream::out | std::fstream::trunc);
    }
    catch (const std::fstream::failure& e)
    {
        std::cout << e.what() << std::endl;
        std::cout << "Exception opening/reading file\n";
        return;
    }

    std::cout << "Writing collected translation strings to " << outFile << std::endl;


    stream << "# This is a translation file for the Saiga Translator." << std::endl;
    stream << "# See saiga/core/util/translator.h for more infos." << std::endl;
    stream << "#" << std::endl;
    stream << "# Structure:  [optional]<required>" << std::endl;
    stream << "# [newentrysymbol]<key><separator><translation><separator>[note]" << std::endl;
    stream << "#" << std::endl;
    stream << "# Special characters:" << std::endl;
    stream << "# <comment>  " << comment << std::endl;
    stream << "# <newentrysymbol>  " << newEntry << std::endl;
    stream << "# <separator>  " << spacer << std::endl;
    stream << "# <escapesymbol>  " << escapesymbol << std::endl;
    stream << "# <newlinesymbol>  " << newlinesymbol << std::endl;
    stream << "#" << std::endl;
    stream << "# Notes:" << std::endl;
    stream << "# - The key has to be unique" << std::endl;
    stream << "# - <separator> and <escapesymbol> in the translation and note has to be escaped by " << escapesymbol
           << std::endl;
    stream << "# - Entries will be sorted alphabetically by the key" << std::endl;
    stream << "# - An optional new entry symbol will be added if the key was previously not defined" << std::endl;
    stream << "# - The note should be used to give a context" << std::endl;
    stream << "# - The file has to be UTF-8 encoded" << std::endl;
    stream << "#" << std::endl;

    for (TranslationEntry& te : entries)
    {
        if (te.newEntry)
        {
            stream << newEntry;
        }
        stream << escapeSpecialCharacters(te.key) << spacer << escapeSpecialCharacters(te.translation) << spacer
               << escapeSpecialCharacters(te.note) << std::endl;
    }
}



std::string Translator::translate(const std::string& key, const std::string& note)
{
    if (translationMap.find(key) == translationMap.end())
    {
        // adding an identity mapping when entry is not present
        std::cerr << "Could not find translation for '" << key << "' in file: " << translationFile << std::endl;
        TranslationEntry te;
        te.newEntry         = true;
        te.key              = key;
        te.translation      = key;
        te.note             = note;
        translationMap[key] = te;
    }

    return translationMap[key].translation;
}

}  // namespace Saiga
