
#pragma once

#include <string>
#include <map>
#include <vector>
#include <saiga/config.h>
using std::string;

class SAIGA_GLOBAL Translator{
private:
    std::string translationFile;

    struct TranslationEntry{
        std::string key;
        std::string translation;
        std::string note;
    };
    char spacer = ',';
    std::map<std::string,TranslationEntry> translationMap;


    void addTranslation(const TranslationEntry& te);
    std::vector<TranslationEntry> getTranslationVector();
    bool readTranslationFile();
public:
    void init(const std::string& file);
    void writeToFile();


    /**
     * @param note note for the translator for the context
     */
    std::string translate(const std::string& key, const std::string &note);


};


SAIGA_GLOBAL extern Translator translator;

namespace translation{
/**
 * @param note note for the translator for the context
 */
SAIGA_GLOBAL inline std::string tr(const std::string& str,const std::string& note = ""){
    return translator.translate(str,note);
}

}


