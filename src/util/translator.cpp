#include "saiga/util/translator.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include "saiga/util/assert.h"

using std::cout;
using std::endl;

Translator translator;


std::vector<std::string> split(const std::string &s, char delim);


std::string Translator::replace(std::string str, const std::string &search, const std::string &replace)
{
    size_t pos = 0;
    while((pos = str.find(search, pos)) != std::string::npos) {
        str.replace(pos, search.length(), replace);
        pos += replace.length();
    }
    return str;
}

std::string Translator::escapeSpecialCharacters(std::string str)
{
    std::string esc(1,escapesymbol);
    str = replace(str,std::string(1,escapesymbol),esc + escapesymbol);
    str = replace(str,std::string(1,spacer),esc + spacer);
    str = replace(str,std::string(1,'\n'),esc + newlinesymbol);
    return str;
}

std::string Translator::unescapeSpecialCharacters(std::string str)
{
    std::string esc(1,escapesymbol);

    size_t pos = 0;
    while((pos = str.find(esc, pos)) != std::string::npos) {
        str.erase(str.begin()+pos);
        char nextChar = str[pos];
        char replacement = nextChar;
        if(nextChar == 'n'){
            replacement = '\n';
        }
        str[pos] = replacement;
        pos++;
    }
    return str;
}

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
        for(std::string line;std::getline(stream, line);lineNumber++) {

            TranslationEntry te;

            if (stream.eof()){
                break;
            }

            //remove carriage return from windows
            line.erase( std::remove(line.begin(), line.end(), '\r'), line.end() );

            utf32string utf32line = Encoding::UTF8toUTF32(line);

            if(utf32line.size() == 0 || utf32line.front() == comment)
                continue;

            //remove new entry symbol
            if(utf32line.front() == newEntry){
                utf32line.erase(utf32line.begin());
                te.newEntry = true;
            }

            utf32line.erase( std::remove(utf32line.begin(), utf32line.end(), '\r'), utf32line.end() );


            //find positions of spacers (without leading escape symbols)
            std::vector<int> spacerPos;
            int pos = 0;
            uint32_t prev = 0;
            for(auto c : utf32line){
                if(c == spacer && prev != escapesymbol){
                    spacerPos.push_back(pos);
                }
                prev = c;
                pos++;
            }
            assert(spacerPos.size() == 2);

            utf32string key(utf32line.begin(),utf32line.begin()+spacerPos[0]);
            utf32string trans(utf32line.begin()+spacerPos[0]+1,utf32line.begin()+spacerPos[1]);
            utf32string note(utf32line.begin()+spacerPos[1]+1,utf32line.end());

            assert(key.size() > 0);

            te.key =            unescapeSpecialCharacters(Encoding::UTF32toUTF8(key));
            te.translation =    unescapeSpecialCharacters(Encoding::UTF32toUTF8(trans));
            te.note =           unescapeSpecialCharacters(Encoding::UTF32toUTF8(note));
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


    stream << "# This is a translation file for the Saiga Translator." << endl;
    stream << "# See saiga/util/translator.h for more infos." << endl;
    stream << "#" << endl;
    stream << "# Structure:  [optional]<required>" << endl;
    stream << "# [newentrysymbol]<key><separator><translation><separator>[note]" << endl;
    stream << "#" << endl;
    stream << "# Special characters:" << endl;
    stream << "# <comment>  " << comment << endl;
    stream << "# <newentrysymbol>  " << newEntry << endl;
    stream << "# <separator>  " << spacer << endl;
    stream << "# <escapesymbol>  " << escapesymbol << endl;
    stream << "# <newlinesymbol>  " << newlinesymbol << endl;
    stream << "#" << endl;
    stream << "# Notes:" << endl;
    stream << "# - The key has to be unique" << endl;
    stream << "# - No special characters in the key" << endl;
    stream << "# - <separator> and <escapesymbol> in the translation and note has to be escaped by " << escapesymbol << endl;
    stream << "# - Entries will be sorted alphabetically by the key" << endl;
    stream << "# - An optional new entry symbol will be added if the key was previously not defined" << endl;
    stream << "# - The note should be used to give a context" << endl;
    stream << "# - The file has to be UTF-8 encoded" << endl;
    stream << "#" << endl;

    for(TranslationEntry& te : entries){
        if(te.newEntry){
            stream << newEntry;
        }
        stream << te.key << spacer <<
                  escapeSpecialCharacters(te.translation)  << spacer  <<
                  escapeSpecialCharacters(te.note) << endl;
    }
}



std::string Translator::translate(const std::string &key,const std::string &note)
{
    if(translationMap.find(key) == translationMap.end()){
        //adding an identity mapping when entry is not present
        std::cerr << "Could not find translation for '" << key << "' in file: " << translationFile << std::endl;
        TranslationEntry te;
        te.newEntry = true;
        te.key = key;
        te.translation = key;
        te.note = note;
        translationMap[key] = te;
    }

    return translationMap[key].translation;
}
