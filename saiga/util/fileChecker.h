#pragma once

#include <saiga/config.h>
#include <string>
#include <iostream>
#include <vector>

class SAIGA_GLOBAL FileChecker{
public:

    FileChecker();

    //searches for 'file' at all search pathes and returns the full name
    std::string getFile(const std::string& file);

    //returns the full file name of 'file' that is relative addressed to 'basefile'
    std::string getRelative(const std::string& baseFile, const std::string& file);

    /**
     * returns the parent directory of 'file'
     * Example:
     * test/image.png
     * ->   test/
     */
    std::string getParentDirectory(const std::string& file);

	/**
	* returns the raw file name of 'file'
	* Example:
	* test/image.png
	* ->   image.png
	*/
	std::string getFileName(const std::string& file);

    void addSearchPath(const std::string& path);

private:
    //all file search functions search at these pathes.
    //the first match will return.
    //the empty path is added by default.
    std::vector<std::string> searchPathes;


    bool existsFile(const std::string& file);
};
