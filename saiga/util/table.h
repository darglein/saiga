#pragma once

#include <saiga/config.h>
#include <vector>
#include <iomanip>

class SAIGA_GLOBAL Table{
private:
    int currentCol = 0;
    int numCols = 0;
    int precision = 6;
    std::vector<int> colWidth;
public:

    Table() {}
    Table(std::vector<int> colWidth);

    void setColWidth(std::vector<int> colWidth);

    void setFloatPrecision(int p){
        precision = p;
    }

    template<typename T>
    Table& operator<<(const T& t){

        std::cout << std::setprecision(precision);
        std::cout << std::setw(colWidth[currentCol]) << std::left << t;
        currentCol = (currentCol + 1) % numCols;
        if(currentCol == 0){
            std::cout << std::endl;
        }
        std::cout.unsetf(std::ios_base::floatfield);
        return *this;
    }

};
