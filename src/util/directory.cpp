#include "saiga/util/directory.h"
#include "saiga/util/assert.h"

#include <iostream>
#include <string>
#include <sys/stat.h>

namespace Saiga {

Directory::Directory(const std::string &dir)
{
    dirname = dir;
    if ((this->dir = opendir (dir.c_str())) == NULL) {
        std::cout<<"could not open directory: "<<dir<<std::endl;
    }
}

Directory::~Directory()
{
    closedir (dir);
}

void Directory::getFiles(std::vector<std::string> &out)
{
    if(!dir)
        return;

    struct dirent *ent;
    /* print all the files and directories within directory */
    while ((ent = readdir (dir)) != NULL) {
        if(ent->d_type == DT_REG)
            out.push_back(std::string(ent->d_name));
    }

}

bool hasEnding (std::string const &fullString, std::string const &ending) {
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

void Directory::getFiles(std::vector<std::string> &out, const std::string &ending)
{
    if(!dir)
        return;

    struct dirent *ent;
    while ((ent = readdir (dir)) != NULL) {
        if(ent->d_type == DT_REG){
            std::string str(ent->d_name);
            if(hasEnding(str,ending))
                out.push_back(str);
        }else if(ent->d_type == DT_UNKNOWN){
            //On some filesystems like XFS the d_type is always DT_UNKNOWN.
            //We need to use stat to check if it's a regular file. (Thanks to Samuel Nelson)
            std::string fullFileName = dirname + "/" + std::string(ent->d_name);
            struct stat st;
            int ret = stat(fullFileName.c_str(), &st);
            SAIGA_ASSERT(ret == 0);
            if(S_ISREG(st.st_mode)){
                std::string str(ent->d_name);
                if(hasEnding(str,ending))
                    out.push_back(str);
            }
        }
    }

}

}
