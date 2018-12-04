/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <sstream>
#include <vector>

namespace Saiga
{
class SAIGA_GLOBAL Console : public std::ostream
{
   private:
    class ConsoleBuffer : public std::stringbuf
    {
       private:
        Console& parent;

       public:
        ConsoleBuffer(Console& parent);
        ~ConsoleBuffer();

        int sync();
    };

    std::vector<std::string> functionList;

    int historyPointer = 0;
    std::vector<std::string> commandHistory;

   public:
    Console();
    virtual ~Console();

    void createFunctionList();
    std::string completeText(const std::string& line);

    void previousCommand(std::string& out);
    void lastCommand(std::string& out);
    virtual void execute(const std::string& line);
    virtual void printLine(const std::string& line);
};

}  // namespace Saiga
