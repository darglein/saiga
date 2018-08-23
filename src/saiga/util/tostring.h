/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <sstream>
#include <string>
#include <vector>

/**
 * This acts as a wrapper for std::to_string for build in data types.
 * For more complex data types the ostream operator << is used to create a string.
 * This allows for a simple conversion of vec3 -> string
 */

namespace Saiga {

inline std::string to_string(int v)                 { return std::to_string(v); }
inline std::string to_string(long v)                { return std::to_string(v); }
inline std::string to_string(long long v)           { return std::to_string(v); }
inline std::string to_string(unsigned v)            { return std::to_string(v); }
inline std::string to_string(unsigned long v)       { return std::to_string(v); }
inline std::string to_string(unsigned long long v)  { return std::to_string(v); }
inline std::string to_string(float v)               { return std::to_string(v); }
inline std::string to_string(double v)              { return std::to_string(v); }
inline std::string to_string(long double v)         { return std::to_string(v); }
inline std::string to_string(const std::string& v)  { return v; }
inline std::string to_string(const char* v)         { return std::string(v); }

template<typename T>
inline std::string to_string(const T& v)
{
    std::stringstream sstream;
    sstream << v;
    return sstream.str();
}


inline float to_float(const std::string& str)   {return std::atof(str.c_str());}
inline double to_double(const std::string& str)   {return std::atof(str.c_str());}
inline int to_int(const std::string& str)   {return std::atoi(str.c_str());}
inline long int to_long(const std::string& str)   {return std::atol(str.c_str());}


SAIGA_GLOBAL std::vector<std::string> split(const std::string &s, char delim);
SAIGA_GLOBAL std::string concat(const std::vector<std::string>  &s, char delim);

SAIGA_GLOBAL std::string leadingZeroString(int number, int characterCount);
SAIGA_GLOBAL bool hasEnding (std::string const &fullString, std::string const &ending);

/**
 * @brief fileEnding
 * Extracts and returns the file ending of a string.
 * Example:
 *
 * "bla.jpg" -> "jpg"
 * "/usr/local/test.asdf" -> "asdf"
 */
SAIGA_GLOBAL std::string fileEnding (std::string const &str);
SAIGA_GLOBAL std::string removeFileEnding (std::string const &str);


}
