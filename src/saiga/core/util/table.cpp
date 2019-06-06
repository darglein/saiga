/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/table.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
Table::Table(std::vector<int> colWidth, std::ostream& strm) : strm(strm)
{
    setColWidth(colWidth);
}

void Table::setColWidth(std::vector<int> colWidth)
{
    this->colWidth = colWidth;
    numCols        = colWidth.size();
}

}  // namespace Saiga
