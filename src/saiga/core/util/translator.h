/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <map>
#include <saiga/core/util/encoding.h>
#include <vector>
#include <string>

namespace Saiga
{
class SAIGA_CORE_API Translator
{
   private:
    std::string translationFile;

    struct TranslationEntry
    {
        std::string key;
        std::string translation;
        std::string note;
        bool newEntry = false;
    };
    char spacer        = ',';
    char newEntry      = '>';
    char comment       = '#';
    char escapesymbol  = '\\';
    char newlinesymbol = 'n';
    std::map<std::string, TranslationEntry> translationMap;

    std::string replace(std::string str, const std::string& search, const std::string& replace);
    std::string escapeSpecialCharacters(std::string str);
    std::string unescapeSpecialCharacters(std::string str);

    void addTranslation(const TranslationEntry& te);
    std::vector<TranslationEntry> getTranslationVector();
    bool readTranslationFile();

   public:
    void init(const std::string& file);
    void writeToFile();

    std::string translate(const std::string& key, const std::string& note);
};


SAIGA_CORE_API extern Translator translator;

namespace translation
{
SAIGA_CORE_API inline std::string tr(const std::string& key, const std::string& note = "")
{
    return translator.translate(key, note);
}

// this is a usefull struct to only translate your strings once in a function.
// Example:
// void test(){
//    static translation::trstring tstr("Hello world!");
//    std::cout << tstr.str << std::endl;
//}
struct SAIGA_CORE_API trstring
{
    std::string str;
    trstring(const std::string& key, const std::string& note = "") { str = tr(key, note); }
};


}  // namespace translation

}  // namespace Saiga
