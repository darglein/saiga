/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/util/assert.h"

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
   public:
    Table() : strm(&std::cout) {}
    Table(const std::vector<int>& colWidth, std::ostream& strm = std::cout, char additional_sep = '\0');

    void setStream(std::ostream& strm) { this->strm = &strm; }

    void setColWidth(std::vector<int> colWidth);

    void setFloatPrecision(int p) { precision = p; }

    template <typename T>
    Table& operator<<(const T& t)
    {
        SAIGA_ASSERT(strm);
        if (precision > 0)
        {
            (*strm) << std::setprecision(precision);
        }

        (*strm) << std::setw(colWidth[currentCol]) << std::left << t;

        currentCol++;


        if (currentCol == numCols)
        {
            currentCol = 0;
            (*strm) << std::endl;
        }
        else
        {
            (*strm) << additional_sep;
        }

        if (precision > 0)
        {
            strm->unsetf(std::ios_base::floatfield);
        }
        return *this;
    }

   private:
    int currentCol = 0;
    int numCols    = 0;
    int precision  = -1;
    std::vector<int> colWidth;
    std::ostream* strm  = nullptr;
    char additional_sep = '\0';
};

}  // namespace Saiga
