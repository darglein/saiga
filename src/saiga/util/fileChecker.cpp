/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/fileChecker.h"

#include <fstream>
#include "internal/noGraphicsAPI.h"

namespace Saiga {

FileChecker::FileChecker()
{
    addSearchPath(".");
}

std::string FileChecker::getFile(const std::string &file)
{
    for(std::string &path : searchPathes){
        std::string fullName = path + "/" + file;
        if(existsFile(fullName))
            return fullName;
    }
    return "";
}

std::string FileChecker::getRelative(const std::string &baseFile, const std::string &file)
{
    //first check at path relative to the parent
    auto parent = getParentDirectory(baseFile);
    std::string relativeName = parent + file;
    if(existsFile(relativeName)){
        return relativeName;
    }
    return getFile(file);
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

std::string FileChecker::getFileName(const std::string &file)
{
	//search last '/' from the end
	for (auto it = file.rbegin(); it != file.rend(); ++it){
		if (*it == '/'){
			auto d = std::distance(it, file.rend());
			return file.substr(d);
		}
	}
	return "";
}
void FileChecker::addSearchPath(const std::string &path)
{
    searchPathes.push_back(path);
}

bool FileChecker::existsFile(const std::string &file)
{
    std::ifstream infile(file);
    return infile.good();
}

}
