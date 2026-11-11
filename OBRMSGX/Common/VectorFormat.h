#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

// Original vector coordinates have exactly one format per dataset. Floating
// point arithmetic uses exact float32 expansions of uint8/IEEE binary16 values.
namespace vector_format {
const uint32_t Float16=1;
const uint32_t UInt8=2;
static_assert(sizeof(float)==4 && std::numeric_limits<float>::is_iec559,
              "IEEE-754 float32 arithmetic is required");

inline size_t element_bytes(uint32_t encoding) {
    return encoding==UInt8?1:encoding==Float16?2:0;
}
inline float decode_half(uint16_t half) {
    uint32_t sign=uint32_t(half&0x8000u)<<16;
    uint32_t fraction=half&0x3ffu, exponent=(half>>10)&31u, bits;
    if (!exponent) {
        if (!fraction) bits=sign;
        else {
            int e=-14;
            while (!(fraction&0x400u)) {fraction<<=1;--e;}
            bits=sign|(uint32_t(e+127)<<23)|((fraction&0x3ffu)<<13);
        }
    } else if (exponent==31) bits=sign|0x7f800000u|(fraction<<13);
    else bits=sign|((exponent+112u)<<23)|(fraction<<13);
    float result;
    std::memcpy(&result,&bits,4);
    return result;
}
// IEEE binary16 round-to-nearest, ties-to-even. No saturation or clipping.
inline uint16_t encode_half(float value) {
    uint32_t bits;
    std::memcpy(&bits,&value,4);
    const uint16_t sign=uint16_t((bits>>16)&0x8000u);
    const uint32_t original_exp=(bits>>23)&255u, fraction=bits&0x7fffffu;
    if (original_exp==255) return uint16_t(sign|0x7c00u|(fraction?0x200u:0));
    int exponent=int(original_exp)-127;
    if (exponent>15) return uint16_t(sign|0x7c00u);
    if (exponent < -25) return sign;
    uint32_t mantissa=fraction;
    if (exponent < -14) {
        mantissa|=0x800000u;
        const unsigned shift=unsigned(-exponent-1);
        const uint32_t truncated=mantissa>>shift;
        const uint32_t remainder=mantissa&((uint32_t(1)<<shift)-1);
        const uint32_t halfway=uint32_t(1)<<(shift-1);
        return uint16_t(sign|uint16_t(truncated+(remainder>halfway || (remainder==halfway && (truncated&1)))));
    }
    uint32_t rounded=(mantissa>>13)+((mantissa&0x1fffu)>0x1000u
                       || ((mantissa&0x1fffu)==0x1000u && ((mantissa>>13)&1)));
    if (rounded==0x400u) {rounded=0;++exponent;}
    if (exponent>15) return uint16_t(sign|0x7c00u);
    return uint16_t(sign|uint16_t((exponent+15)<<10)|uint16_t(rounded));
}
inline float coordinate(const uint8_t* data,size_t index,uint32_t encoding) {
    if (encoding==UInt8) return static_cast<float>(data[index]);
    uint16_t half;
    std::memcpy(&half,data+2*index,2);
    return decode_half(half);
}
} // namespace vector_format
