#pragma once

#include "saiga/config.h"

#include <sstream>
#include <string>

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

}
