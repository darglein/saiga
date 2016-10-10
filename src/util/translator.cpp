#include "saiga/util/translator.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "saiga/util/assert.h"

using std::cout;
using std::endl;

Translator translator;


std::vector<std::string> split(const std::string &s, char delim);


void Translator::addTranslation(const Translator::TranslationEntry &te)
{
    if(translationMap.find(te.key) != translationMap.end()){
        std::cerr << "Key " << te.key << " is already in the translation map." << std::endl;
        assert(0);
    }
    translationMap[te.key] = te;
}

std::vector<Translator::TranslationEntry> Translator::getTranslationVector()
{
    std::vector<TranslationEntry> erg;
    for(auto iterator = translationMap.begin(); iterator != translationMap.end(); iterator++) {
        TranslationEntry te = iterator->second;
        erg.push_back(te);
    }
    std::sort(erg.begin(),erg.end(),
              [](TranslationEntry a, TranslationEntry b) -> bool{
        return a.key < b.key;
    }
    );
    return erg;
}

void Translator::init(const std::string &file)
{
    translationFile = file;
    translationMap.clear();

    readTranslationFile();
}

void Translator::writeToFile()
{
    std::string outFile = translationFile;
    std::vector<TranslationEntry> entries = getTranslationVector();
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

    for(TranslationEntry& te : entries){
        stream << te.key << spacer << te.translation  << spacer  << te.note << endl;
    }
}


bool Translator::readTranslationFile()
{
    std::cout << "Reading translation file " << translationFile << endl;

    std::fstream stream;

    try {
        stream.open (translationFile, std::fstream::in);
        if (!stream.good()){
            cout << "Warning Translator: file does not exist! " + translationFile << endl;
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
       for(string line;std::getline(stream, line);lineNumber++) {
            if (stream.eof()){
                break;
            }
            std::vector<std::string> linesplit = split(line,spacer);
            if(linesplit.size() == 2){
                //add empty note
                linesplit.push_back("");
            }
            assert(linesplit.size() == 3);
            TranslationEntry te;
            te.key = linesplit[0];
            te.translation = linesplit[1];
            te.note = linesplit[2];
            cout << lineNumber << " " << te.key << spacer << te.translation << spacer << te.note << endl;
            addTranslation(te);
        }
    }
    catch (const std::fstream::failure &e) {
        cout<< e.what() << std::endl;
        return false;
    }
    return true;
}


std::string Translator::translate(const std::string &key,const std::string &note)
{
     if(translationMap.find(key) == translationMap.end()){
         //adding an identity mapping when entry is not present
         std::cerr << "Could not find translation for '" << key << "' in file: " << translationFile << std::endl;
         TranslationEntry te;
         te.key = key;
         te.translation = key;
         te.note = note;
         translationMap[key] = te;
     }

    return translationMap[key].translation;
}
