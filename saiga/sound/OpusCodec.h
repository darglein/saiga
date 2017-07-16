// (c) Seth Heeren 2013
//
// Based on src/opus_demo.c in opus-1.0.2
// License see http://www.opus-codec.org/license/

#pragma once

#include <stdexcept>
#include <iosfwd>
#include <vector>
#include <memory>

namespace Saiga {

struct OpusErrorException : public virtual std::exception
{
    OpusErrorException(int code) : code(code) {}
    const char* what();
private:
    const int code;
};

struct COpusCodec
{
    COpusCodec(int32_t sampling_rate, int channels);
    ~COpusCodec();

    std::vector<unsigned char> decode_frame(std::istream& fin);
private:
    struct Impl;
    std::unique_ptr<Impl> _pimpl;
};

}
