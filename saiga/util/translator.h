
#pragma once

#include <string>
#include <map>
#include <vector>

using std::string;

class Translator{
private:
    bool isCollecting = false;
    std::map<std::string,std::string> translations;

    std::vector<std::pair<string,string>> collectedStrings;
public:
    void readTranslationFile(const std::string& file);
    void startCollecting();
    void writeToFile();
    /**
     * @param note note for the translator for the context
     */
    std::string translate(const std::string& str, const std::string &note);

    void collect(const std::string& str, const std::string &note);

};


extern Translator* translator;

namespace translation{
/**
 * @param note note for the translator for the context
 */
inline std::string tr(const std::string& str,const std::string& note = ""){
    return translator->translate(str,note);
}

inline void addTranslation(const std::string& str,const std::string& note = ""){
    translator->collect(str,note);
}
}


