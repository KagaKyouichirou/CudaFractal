#include "FixedPoint8U30Naive.h"

FixedPoint8U30Naive::FixedPoint8U30Naive(): data{} {}

std::array<uint8_t, 30> FixedPoint8U30Naive::intoBytes() const
{
    return {
        static_cast<uint8_t>(data[7] >> 22),
        static_cast<uint8_t>(data[7] >> 14),
        static_cast<uint8_t>(data[7] >> 6),
        static_cast<uint8_t>(data[7] << 2 | 3 & data[6] >> 28),
        static_cast<uint8_t>(data[6] >> 20),
        static_cast<uint8_t>(data[6] >> 12),
        static_cast<uint8_t>(data[6] >> 4),
        static_cast<uint8_t>(data[6] << 4 | 15 & data[5] >> 26),
        static_cast<uint8_t>(data[5] >> 18),
        static_cast<uint8_t>(data[5] >> 10),
        static_cast<uint8_t>(data[5] >> 2),
        static_cast<uint8_t>(data[5] << 6 | 63 & data[4] >> 24),
        static_cast<uint8_t>(data[4] >> 16),
        static_cast<uint8_t>(data[4] >> 8),
        static_cast<uint8_t>(data[4]),
        static_cast<uint8_t>(data[3] >> 22),
        static_cast<uint8_t>(data[3] >> 14),
        static_cast<uint8_t>(data[3] >> 6),
        static_cast<uint8_t>(data[3] << 2 | 3 & data[2] >> 28),
        static_cast<uint8_t>(data[2] >> 20),
        static_cast<uint8_t>(data[2] >> 12),
        static_cast<uint8_t>(data[2] >> 4),
        static_cast<uint8_t>(data[2] << 4 | 15 & data[1] >> 26),
        static_cast<uint8_t>(data[1] >> 18),
        static_cast<uint8_t>(data[1] >> 10),
        static_cast<uint8_t>(data[1] >> 2),
        static_cast<uint8_t>(data[1] << 6 | 63 & data[0] >> 24),
        static_cast<uint8_t>(data[0] >> 16),
        static_cast<uint8_t>(data[0] >> 8),
        static_cast<uint8_t>(data[0])
    };
}

FixedPoint8U30Naive FixedPoint8U30Naive::fromBytes(std::array<uint8_t, 30> const& bytes, bool negative)
{
    return FixedPoint8U30Naive(
        // clang-format off
          static_cast<uint32_t>(bytes[29])
        | static_cast<uint32_t>(bytes[28]) << 8
        | static_cast<uint32_t>(bytes[27]) << 16
        | static_cast<uint32_t>(bytes[26] & 63) << 24,

          static_cast<uint32_t>(bytes[26]) >> 6
        | static_cast<uint32_t>(bytes[25]) << 2
        | static_cast<uint32_t>(bytes[24]) << 10
        | static_cast<uint32_t>(bytes[23]) << 18
        | static_cast<uint32_t>(bytes[22] & 15) << 26,

          static_cast<uint32_t>(bytes[22]) >> 4
        | static_cast<uint32_t>(bytes[21]) << 4
        | static_cast<uint32_t>(bytes[20]) << 12
        | static_cast<uint32_t>(bytes[19]) << 20
        | static_cast<uint32_t>(bytes[18] & 3) << 28,

          static_cast<uint32_t>(bytes[18]) >> 2
        | static_cast<uint32_t>(bytes[17]) << 6
        | static_cast<uint32_t>(bytes[16]) << 14
        | static_cast<uint32_t>(bytes[15]) << 22,

          static_cast<uint32_t>(bytes[14])
        | static_cast<uint32_t>(bytes[13]) << 8
        | static_cast<uint32_t>(bytes[12]) << 16
        | static_cast<uint32_t>(bytes[11] & 63) << 24,
    
          static_cast<uint32_t>(bytes[11]) >> 6
        | static_cast<uint32_t>(bytes[10]) << 2
        | static_cast<uint32_t>(bytes[9]) << 10
        | static_cast<uint32_t>(bytes[8]) << 18
        | static_cast<uint32_t>(bytes[7] & 15) << 26,

          static_cast<uint32_t>(bytes[7]) >> 4
        | static_cast<uint32_t>(bytes[6]) << 4
        | static_cast<uint32_t>(bytes[5]) << 12
        | static_cast<uint32_t>(bytes[4]) << 20
        | static_cast<uint32_t>(bytes[3] & 3) << 28,

          static_cast<uint32_t>(bytes[3]) >> 2
        | static_cast<uint32_t>(bytes[2]) << 6
        | static_cast<uint32_t>(bytes[1]) << 14
        | static_cast<uint32_t>(bytes[0]) << 22
        | static_cast<uint32_t>(negative) << 31
        // clang-format on
    );
}

__host__ __device__ void FixedPoint8U30Naive::flip()
{
    data[7] ^= 0x80000000;
}

template <size_t idx>
__host__ __device__ static inline void carryCore(uint32_t* data)
{
    data[idx + 1] += (data[idx] & 0xC0000000) >> 30;
    data[idx] &= 0x3FFFFFFF;
}

template <size_t... Indices>
__host__ __device__ static inline void carryCoreSeq(uint32_t* data, std::index_sequence<Indices...>)
{
    (carryCore<Indices>(data), ...);
}

__host__ __device__ static inline void carryCoreAll(uint32_t* data)
{
    carryCoreSeq(data, std::make_index_sequence<7>{});
    data[7] &= 0x3FFFFFFF;
}

__host__ __device__ static inline void complementEle(uint32_t& e)
{
    e = 0x3FFFFFFF - e;
}

template <size_t... Indices>
__host__ __device__ static inline void complementSeq(uint32_t* data, std::index_sequence<Indices...>)
{
    (complementEle(data[Indices]), ...);
}

__host__ __device__ static inline void complement(uint32_t* data)
{
    // assumes that data[7] doesnt include the sign bit
    complementSeq(data, std::make_index_sequence<8>{});
    data[0]++;
    carryCoreAll(data);
}

__host__ __device__ void FixedPoint8U30Naive::add(FixedPoint8U30Naive const& other)
{
    // assumes that the two instances involed are not the same one
    uint32_t const signX = data[7] & 0x80000000;
    data[7] &= 0x3FFFFFFF;
    uint32_t const signY = other.data[7] & 0x80000000;
    if (signX == signY) {
        data[0] += other.data[0];
        data[1] += other.data[1];
        data[2] += other.data[2];
        data[3] += other.data[3];
        data[4] += other.data[4];
        data[5] += other.data[5];
        data[6] += other.data[6];
        data[7] += other.data[7] & 0x3FFFFFFF;
        carryCoreAll(data);
        data[7] |= signX;
    } else {
        if (signX > 0) {
            data[0] = 0x3FFFFFFF - data[0] + other.data[0];
            data[1] = 0x3FFFFFFF - data[1] + other.data[1];
            data[2] = 0x3FFFFFFF - data[2] + other.data[2];
            data[3] = 0x3FFFFFFF - data[3] + other.data[3];
            data[4] = 0x3FFFFFFF - data[4] + other.data[4];
            data[5] = 0x3FFFFFFF - data[5] + other.data[5];
            data[6] = 0x3FFFFFFF - data[6] + other.data[6];
            data[7] = 0x3FFFFFFF - data[7] + (other.data[7] & 0x3FFFFFFF);
        } else {
            data[0] += 0x3FFFFFFF - other.data[0];
            data[1] += 0x3FFFFFFF - other.data[1];
            data[2] += 0x3FFFFFFF - other.data[2];
            data[3] += 0x3FFFFFFF - other.data[3];
            data[4] += 0x3FFFFFFF - other.data[4];
            data[5] += 0x3FFFFFFF - other.data[5];
            data[6] += 0x3FFFFFFF - other.data[6];
            data[7] += 0x3FFFFFFF - (other.data[7] & 0x3FFFFFFF);
        }
        data[0]++;
        carryCoreAll(data);
        if (data[7] >= 0x20000000) {
            complement(data);
            data[7] |= 0x80000000;
        }
    }
}

__host__ __device__ void FixedPoint8U30Naive::dou()
{
    uint32_t const sign = data[7] & 0x80000000;
    data[7] &= 0x3FFFFFFF;
    data[0] <<= 1;
    data[1] <<= 1;
    data[2] <<= 1;
    data[3] <<= 1;
    data[4] <<= 1;
    data[5] <<= 1;
    data[6] <<= 1;
    data[7] <<= 1;
    carryCoreAll(data);
    data[7] |= sign;
}

template <size_t idx>
__host__ __device__ static inline void mulVecToNum(uint64_t* mid, uint32_t const* x, uint64_t p)
{
    mid[idx + 0] += p * x[0];
    mid[idx + 1] += p * x[1];
    mid[idx + 2] += p * x[2];
    mid[idx + 3] += p * x[3];
    mid[idx + 4] += p * x[4];
    mid[idx + 5] += p * x[5];
    mid[idx + 6] += p * x[6];
    mid[idx + 7] += p * x[7];
}

// clang-format off
template <size_t... Indices>
__host__ __device__ static inline void mulVecToVec(
    uint64_t* mid,
    uint32_t const* x,
    uint32_t const* y,
    std::index_sequence<Indices...>
) {
    (mulVecToNum<Indices>(mid, x, static_cast<uint64_t>(y[Indices])), ...);
}
// clang-format on

template <size_t idx>
__host__ __device__ static inline void carryMid(uint64_t* mid)
{
    mid[idx + 1] += (mid[idx] & 0xFFFFFFFFC0000000) >> 30;
    mid[idx] &= 0x3FFFFFFF;
}

template <size_t... Indices>
__host__ __device__ static inline void carryMidSeq(uint64_t* mid, std::index_sequence<Indices...>)
{
    (carryMid<Indices>(mid), ...);
}

__host__ __device__ void FixedPoint8U30Naive::mul(FixedPoint8U30Naive& other)
{
    // assumes that the two instances involed are not the same one
    uint32_t const signX = data[7] & 0x80000000;
    data[7] &= 0x3FFFFFFF;
    uint32_t const signY = other.data[7] & 0x80000000;
    other.data[7] &= 0x3FFFFFFF;

    uint64_t mid[16]{};

    mulVecToVec(mid, data, other.data, std::make_index_sequence<8>{});

    carryMidSeq(mid, std::make_index_sequence<7>{});
    // rounding
    mid[7] += mid[6] >= 0x20000000;
    carryMidSeq(mid, std::index_sequence<7, 8, 9, 10, 11, 12, 13>{});
    mid[14] &= 0x3FFFFFFF;

    data[0] = static_cast<uint32_t>(mid[7]);
    data[1] = static_cast<uint32_t>(mid[8]);
    data[2] = static_cast<uint32_t>(mid[9]);
    data[3] = static_cast<uint32_t>(mid[10]);
    data[4] = static_cast<uint32_t>(mid[11]);
    data[5] = static_cast<uint32_t>(mid[12]);
    data[6] = static_cast<uint32_t>(mid[13]);
    data[7] = static_cast<uint32_t>(mid[14]) | (signX ^ signY);

    other.data[7] |= signY;
}

__host__ __device__ void FixedPoint8U30Naive::mul(unsigned int f)
{
    // assumes that the instance is non-negative
    uint64_t c = 0;
    c += static_cast<uint64_t>(data[0]) * f;
    data[0] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[1]) * f;
    data[1] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[2]) * f;
    data[2] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[3]) * f;
    data[3] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[4]) * f;
    data[4] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[5]) * f;
    data[5] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[6]) * f;
    data[6] = c & 0x3FFFFFFF;
    c >>= 30;
    c += static_cast<uint64_t>(data[7]) * f;
    data[7] = c & 0x3FFFFFFF;
}

__host__ __device__ void FixedPoint8U30Naive::sqr()
{
    data[7] &= 0x3FFFFFFF;
    uint64_t mid[16]{};

    mulVecToVec(mid, data, data, std::make_index_sequence<8>{});

    carryMidSeq(mid, std::make_index_sequence<7>{});
    // rounding
    mid[7] += mid[6] >= 0x20000000;
    carryMidSeq(mid, std::index_sequence<7, 8, 9, 10, 11, 12, 13>{});
    mid[14] &= 0x3FFFFFFF;

    data[0] = static_cast<uint32_t>(mid[7]);
    data[1] = static_cast<uint32_t>(mid[8]);
    data[2] = static_cast<uint32_t>(mid[9]);
    data[3] = static_cast<uint32_t>(mid[10]);
    data[4] = static_cast<uint32_t>(mid[11]);
    data[5] = static_cast<uint32_t>(mid[12]);
    data[6] = static_cast<uint32_t>(mid[13]);
    data[7] = static_cast<uint32_t>(mid[14]);
}

bool FixedPoint8U30Naive::reasonable() const
{
    return data[7] < 4;
}

bool FixedPoint8U30Naive::zero() const
{
    return 0 == (data[0] | data[1] | data[2] | data[3] | data[4] | data[5] | data[6] | data[7]);
}

bool FixedPoint8U30Naive::exceeds(FixedPoint8U30Naive const& bound)
{
    if (data[7] < bound.data[7]) {
        return false;
    } else if (data[7] > bound.data[7]) {
        return true;
    }
    if (data[6] < bound.data[6]) {
        return false;
    } else if (data[6] > bound.data[6]) {
        return true;
    }
    if (data[5] < bound.data[5]) {
        return false;
    } else if (data[5] > bound.data[5]) {
        return true;
    }
    if (data[4] < bound.data[4]) {
        return false;
    } else if (data[4] > bound.data[4]) {
        return true;
    }
    if (data[3] < bound.data[3]) {
        return false;
    } else if (data[3] > bound.data[3]) {
        return true;
    }
    if (data[2] < bound.data[2]) {
        return false;
    } else if (data[2] > bound.data[2]) {
        return true;
    }
    if (data[1] < bound.data[1]) {
        return false;
    } else if (data[1] > bound.data[1]) {
        return true;
    }
    if (data[0] < bound.data[0]) {
        return false;
    } else if (data[0] > bound.data[0]) {
        return true;
    }
    return false;
}

__host__ __device__ bool FixedPoint8U30Naive::checkNorm(FixedPoint8U30Naive const& x2, FixedPoint8U30Naive const& y2)
{
    auto tmp = x2;
    tmp.add(y2);
    return tmp.data[7] < 4;
}

constexpr FixedPoint8U30Naive::FixedPoint8U30Naive(
    uint32_t d0,
    uint32_t d1,
    uint32_t d2,
    uint32_t d3,
    uint32_t d4,
    uint32_t d5,
    uint32_t d6,
    uint32_t d7
):
    data{d0, d1, d2, d3, d4, d5, d6, d7}
{}