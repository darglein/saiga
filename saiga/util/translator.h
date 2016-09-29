
#pragma once

#include <string>
#include <map>
#include <vector>
#include <saiga/config.h>
using std::string;

class SAIGA_GLOBAL Translator{
private:
    bool isCollecting = false;
    std::map<std::string,std::string> translations;

    std::vector<std::pair<string,string>> collectedStrings;
public:
    bool readTranslationFile(const std::string& file);
    void startCollecting();
    void writeToFile();
    /**
     * @param note note for the translator for the context
     */
    std::string translate(const std::string& str, const std::string &note);

    void collect(const std::string& str, const std::string &note);

};


SAIGA_GLOBAL extern Translator* translator;

namespace translation{
/**
 * @param note note for the translator for the context
 */
SAIGA_GLOBAL inline std::string tr(const std::string& str,const std::string& note = ""){
    return translator->translate(str,note);
}

SAIGA_GLOBAL inline void addTranslation(const std::string& str,const std::string& note = ""){
    translator->collect(str,note);
}
}


