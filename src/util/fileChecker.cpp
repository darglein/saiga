#include "saiga/util/fileChecker.h"

std::string FileChecker::getRelative(const std::string &baseFile, const std::string &file)
{
    auto parent = getParentDirectory(baseFile);
    return parent + file;
}

std::string FileChecker::getParentDirectory(const std::string &file)
{
    //search last '/' from the end
    for(auto it = file.rbegin() ; it != file.rend(); ++it){
        if(*it == '/'){
            auto d = std::distance(it,file.rend());
            return file.substr(0,d);
        }
    }
    return "";
}
