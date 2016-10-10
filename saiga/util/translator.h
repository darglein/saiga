
#pragma once

#include <string>
#include <map>
#include <vector>
#include <iostream>
#include <saiga/config.h>

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

    std::string translate(const std::string& key, const std::string &note);
};


SAIGA_GLOBAL extern Translator translator;

namespace translation{


SAIGA_GLOBAL inline std::string tr(const std::string& key,const std::string& note = ""){
    return translator.translate(key,note);
}

//this is a usefull struct to only translate your strings once in a function.
//Example:
//void test(){
//    static translation::trstring tstr("Hello world!");
//    cout << tstr.str << endl;
//}
struct SAIGA_GLOBAL trstring{
    std::string str;
    trstring(const std::string& key, const std::string& note = ""){
        str = tr(key,note);
    }
};


}


