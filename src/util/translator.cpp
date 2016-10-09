#include "saiga/util/translator.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "assert.h"

using std::cout;
using std::endl;

Translator* translator = nullptr;


bool Translator::readTranslationFile(const std::string &file)
{
    assert(!isCollecting);

    std::fstream stream;

    try {
        stream.open (file, std::fstream::in);
        if (!stream.good()){
            cout << "Warning Translator: file does not exist! " + file << endl;
            return false;
        }

    }
    catch (const std::fstream::failure &e) {
        cout<< e.what() << endl;
        cout << "Exception opening/reading file\n";
        return false;
    }

    int lineNumber = 0;
    try {

        string line;
       for(;;) {
            std::getline(stream, line);

//            cout << "line: " << line << endl;

            int l = (int)line.find_last_of(",");
            if (l == (int)string::npos){
                cout << "corrupt translation file for line " << line << endl;
                break;
            }

            std::string english = line.substr(0,l);
            std::string translated = line.substr(l + 1);

            //ignore first line
            if (lineNumber == 0){
                //does not work if string is encoded in another format
                if (english == "English"){
                    continue;
                } else {
                    cout << "Warning:  invalid first line in translation file!" << endl;
                }
            }

            cout << "tr: " << english << ":" << translated << endl;

            if (translations.find(english) != translations.end()){
                if (translations[english] == translated){
                    cout << "warning, duplicate entry for: " << english << endl;
                } else{
                    assert(false && "Duplicate, mismatching entry for translation file!");
                }
            }


            translations[english] = translated;
//            entries.push_back(entry);


            if (stream.eof()){
                break;
            }


            lineNumber++;
        }
    }
    catch (const std::fstream::failure &e) {
        cout<< e.what() << std::endl;
        return false;
    }
    return true;
}

void Translator::startCollecting()
{

#if defined(SAIGA_RELEASE)
    return;
#endif

    isCollecting = true;
    collectedStrings.clear();
    collectedStrings.push_back(std::make_pair("English","The language of this translation file"));
}

void Translator::writeToFile()
{

#if defined(SAIGA_RELEASE)
    return;
#endif

    if (!isCollecting)
        return;

    std::string outFile = "lang/collected.txt";

    std::fstream stream;

    try {
        stream.open (outFile,  std::fstream::out | std::fstream::trunc);
    }
    catch (const std::fstream::failure &e) {
        cout<< e.what() << endl;
        cout << "Exception opening/reading file\n";
        return;
    }

    std::cout << "Writing collected translation strings to " << outFile << std::endl;

    for(std::pair<string,string>& p : collectedStrings){
        stream << p.first << "," << p.second << endl;
    }
}

std::string Translator::translate(const std::string &str,const std::string &note)
{
    if (isCollecting){
        collect(str,note);
        return str;
    } else {
        assert(translations.find(str) != translations.end());
        return translations[str];
    }
}

void Translator::collect(const std::string &str, const std::string &note)
{
    if (!isCollecting)
        return;

    bool found = false;
    for(auto& p : collectedStrings){
        if (p.first == str){
            //     cout << "warning: " << str << " already in collected list" << endl;
            found = true;
            if (p.second == ""){
                p.second = note;
            } else {
#if defined(SAIGA_DEBUG)
                if (note != p.second){
                    cout << "Warning: " << "different notes for translated string: " << str << endl;
                }
#endif
            }
            break;
        }
    }

    if (!found){
        collectedStrings.push_back(std::make_pair(str,note));
    }
}
