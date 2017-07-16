#include "saiga/rendering/overlay/console.h"
#include "saiga/util/inputcontroller.h"

namespace Saiga {

Console::ConsoleBuffer::ConsoleBuffer(Console& parent):parent(parent){
}

Console::ConsoleBuffer::~ConsoleBuffer(){
    pubsync();
}

int Console::ConsoleBuffer::sync() {
    parent.printLine(str());
    str("");
    return !std::cout;
}


Console::Console() : std::ostream(new ConsoleBuffer(*this)){
}

Console::~Console() {
    delete rdbuf();
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::stringstream ss(s);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}



void Console::execute(const std::string &line){


    if(!IC.execute(line))
        return;

    commandHistory.push_back(line);
    historyPointer = commandHistory.size()-1;
}

void Console::lastCommand(std::string &out){

    if(historyPointer>=0 && commandHistory.size()>0){
        out = commandHistory[historyPointer--];
    }

    if(historyPointer<0){
        historyPointer = commandHistory.size()-1;
    }
}


void Console::previousCommand(std::string &out){
    historyPointer++;
    if(historyPointer<(int)commandHistory.size()){
        out = commandHistory[historyPointer];
    }else{
        historyPointer = 0;
    }
}

void Console::printLine(const std::string &line){
    (void)line;
}

void Console::createFunctionList(){
    functionList.resize(0);
    IC.functionList(functionList);
}

std::string Console::completeText(const std::string &line){
    std::string erg("");
    //search function list for right range
    auto start = functionList.end();
    auto end = functionList.end();
    bool found = false;
    for(auto iter=functionList.begin(); iter!=functionList.end(); ++iter){
        std::string &str = *iter;
        if(!found && str.substr(0,line.size()) == line){
            start = iter;
            found = true;
        }
        if(found && str.substr(0,line.size()) != line){
            end = iter;
            break;
        }
    }

    int count = (end-start);
    if(count==0){
        return line;
    }
    if(count==1){
        return *start;
    }
    //print first few commands
    int i=0;
    for(auto iter=start; iter!=end; ++iter){
        if(i>=5){
            *this<<"...";
            break;
        }
        *this<<(*iter)<<" ";
        i++;
    }
    *this<<std::endl;


    auto res = std::mismatch((*start).begin(), (*start).end(), (*(end-1)).begin());
    int c = (res.first-(*start).begin());

    return (*start).substr(0,c);

}

}
