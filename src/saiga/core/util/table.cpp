/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/util/table.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
Table::Table(const std::vector<int>& colWidth, std::ostream& strm, char additional_sep)
    : strm(&strm), additional_sep(additional_sep)
{
    setColWidth(colWidth);
}

void Table::setColWidth(std::vector<int> colWidth)
{
    this->colWidth = colWidth;
    numCols        = colWidth.size();
}

}  // namespace Saiga
