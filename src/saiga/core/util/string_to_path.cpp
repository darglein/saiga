#include "string_to_path.h"



bool is_valid_utf8(const std::string& string)
{
    const unsigned char* bytes = (const unsigned char*)string.c_str();
    unsigned int cp;
    int num;

    while (*bytes != 0x00)
    {
        if ((*bytes & 0x80) == 0x00)
        {
            // U+0000 to U+007F 
            cp = (*bytes & 0x7F);
            num = 1;
        }
        else if ((*bytes & 0xE0) == 0xC0)
        {
            // U+0080 to U+07FF 
            cp = (*bytes & 0x1F);
            num = 2;
        }
        else if ((*bytes & 0xF0) == 0xE0)
        {
            // U+0800 to U+FFFF 
            cp = (*bytes & 0x0F);
            num = 3;
        }
        else if ((*bytes & 0xF8) == 0xF0)
        {
            // U+10000 to U+10FFFF 
            cp = (*bytes & 0x07);
            num = 4;
        }
        else
        {
            return false;
        }

        bytes += 1;
        for (int i = 1; i < num; ++i)
        {
            if ((*bytes & 0xC0) != 0x80)
            {
                return false;
            }
            cp = (cp << 6) | (*bytes & 0x3F);
            bytes += 1;
        }

        if ((cp > 0x10FFFF) ||
            ((cp >= 0xD800) && (cp <= 0xDFFF)) ||
            ((cp <= 0x007F) && (num != 1)) ||
            ((cp >= 0x0080) && (cp <= 0x07FF) && (num != 2)) ||
            ((cp >= 0x0800) && (cp <= 0xFFFF) && (num != 3)) ||
            ((cp >= 0x10000) && (cp <= 0x1FFFFF) && (num != 4)))
        {
            return false;
        }
    }

    return true;
}

std::filesystem::path string_to_path(const std::string& s)
{
    return is_valid_utf8(s) ? std::filesystem::u8path(s) : std::filesystem::path(s);
}

std::string path_to_string(const std::filesystem::path& s)
{
    try
    {
        return s.string();
    }
    catch (const std::filesystem::filesystem_error&)
    {
        return {};
    }
    catch (const std::system_error&)
    {
        return {};
    }
}

