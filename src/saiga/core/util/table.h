/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <iomanip>
#include <iostream>
#include <vector>

namespace Saiga
{
/**
 * A formated table for std::out.
 *
 * Usage:
 *
 * Saiga::Table table({6,7,7});
 *
 * table << "Id" << "Before" << "After";
 * for(int i = 0; i < N; ++i)
 * {
 *      table << i << h_data[i] << res[i];
 * }
 */
class SAIGA_CORE_API Table
{
   private:
    int currentCol = 0;
    int numCols    = 0;
    int precision  = 6;
    std::vector<int> colWidth;
    std::ostream& strm;

   public:
    Table() : strm(std::cout) {}
    Table(std::vector<int> colWidth, std::ostream& strm = std::cout);

    void setColWidth(std::vector<int> colWidth);

    void setFloatPrecision(int p) { precision = p; }

    template <typename T>
    Table& operator<<(const T& t)
    {
        strm << std::setprecision(precision);
        strm << std::setw(colWidth[currentCol]) << std::left << t;
        currentCol = (currentCol + 1) % numCols;
        if (currentCol == 0)
        {
            strm << std::endl;
        }
        strm.unsetf(std::ios_base::floatfield);
        return *this;
    }
};

}  // namespace Saiga
