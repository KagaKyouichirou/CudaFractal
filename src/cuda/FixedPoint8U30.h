#pragma once

#include <cuda_runtime.h>
#include <array>

/*
8 30-bit digits, each stored in uint32
*/
class FixedPoint8U30
{
public:
    /* with uninitialized data */
    __host__ __device__ __forceinline__ explicit FixedPoint8U30() {}
    ~FixedPoint8U30() = default;

    std::array<uint8_t, 30> intoBytes() const;
    static FixedPoint8U30 fromBytes(std::array<uint8_t, 30> const& bytes);

    bool reasonable() const;
    bool nonzero() const;
    bool exceeds(FixedPoint8U30 const& bound) const;
    uint32_t const* raw() const;

    /* self = 2's complement of src; no carrying afterwards */
    __host__ __device__ __forceinline__ void flip(FixedPoint8U30 const& src)
    {
        data[0] = 0x40000000 - src.data[0];
        data[1] = 0x3FFFFFFF - src.data[1];
        data[2] = 0x3FFFFFFF - src.data[2];
        data[3] = 0x3FFFFFFF - src.data[3];
        data[4] = 0x3FFFFFFF - src.data[4];
        data[5] = 0x3FFFFFFF - src.data[5];
        data[6] = 0x3FFFFFFF - src.data[6];
        data[7] = 0x3FFFFFFF - src.data[7];
    }

    /* 2's complement; no carrying afterwards */
    __host__ __device__ __forceinline__ void flip()
    {
        flip(*this);
    }

    /* no carrying afterwards */
    __host__ __device__ __forceinline__ void add(FixedPoint8U30 const& src)
    {
        data[0] += src.data[0];
        data[1] += src.data[1];
        data[2] += src.data[2];
        data[3] += src.data[3];
        data[4] += src.data[4];
        data[5] += src.data[5];
        data[6] += src.data[6];
        data[7] += src.data[7];
    }

    /* multiply an integer */
    __host__ __device__ __forceinline__ void mul(uint64_t f)
    {
        uint64_t c = f * data[0];
        data[0] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[1];
        data[1] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[2];
        data[2] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[3];
        data[3] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[4];
        data[4] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[5];
        data[5] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[6];
        data[6] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
        c >>= 30;
        c += f * data[7];
        data[7] = static_cast<uint32_t>(c) & 0x3FFFFFFF;
    }

    /* zero-initialized instance */
    __device__ __forceinline__ static FixedPoint8U30 zero()
    {
        return FixedPoint8U30(0, 0, 0, 0, 0, 0, 0, 0);
    }

    /* finish carrying */
    __device__ __forceinline__ void carry()
    {
        data[1] += data[0] >> 30;
        data[2] += data[1] >> 30;
        data[3] += data[2] >> 30;
        data[4] += data[3] >> 30;
        data[5] += data[4] >> 30;
        data[6] += data[5] >> 30;
        data[7] += data[6] >> 30;

        data[0] &= 0x3FFFFFFF;
        data[1] &= 0x3FFFFFFF;
        data[2] &= 0x3FFFFFFF;
        data[3] &= 0x3FFFFFFF;
        data[4] &= 0x3FFFFFFF;
        data[5] &= 0x3FFFFFFF;
        data[6] &= 0x3FFFFFFF;
        data[7] &= 0x3FFFFFFF;
    }

    /* return the original sign */
    __device__ __forceinline__ bool abs()
    {
        bool sign = data[7] >= 0x20000000;
        if (sign) {
            flip();
            carry();
        }
        return sign;
    }

    /* self = src * src */
    __device__ __forceinline__ void sqr(FixedPoint8U30 const& src)
    {
        uint64_t mid[15]{};
#define SQR_ACC(idx)                                               \
    {                                                              \
        uint64_t const key = static_cast<uint64_t>(src.data[idx]); \
        mid[idx + 0] += key * src.data[0];                         \
        mid[idx + 1] += key * src.data[1];                         \
        mid[idx + 2] += key * src.data[2];                         \
        mid[idx + 3] += key * src.data[3];                         \
        mid[idx + 4] += key * src.data[4];                         \
        mid[idx + 5] += key * src.data[5];                         \
        mid[idx + 6] += key * src.data[6];                         \
        mid[idx + 7] += key * src.data[7];                         \
    }
        SQR_ACC(0);
        SQR_ACC(1);
        SQR_ACC(2);
        SQR_ACC(3);
        SQR_ACC(4);
        SQR_ACC(5);
        SQR_ACC(6);
        SQR_ACC(7);
#undef SQR_ACC
        retrieve(mid);
    }

    /* self = 2 * lhs * rhs */
    __device__ __forceinline__ void dmul(FixedPoint8U30 const& lhs, FixedPoint8U30 const& rhs)
    {
        uint64_t mid[15]{};
#define DMUL_ACC(idx)                                                   \
    {                                                                   \
        uint64_t const key = static_cast<uint64_t>(lhs.data[idx] << 1); \
        mid[idx + 0] += key * rhs.data[0];                              \
        mid[idx + 1] += key * rhs.data[1];                              \
        mid[idx + 2] += key * rhs.data[2];                              \
        mid[idx + 3] += key * rhs.data[3];                              \
        mid[idx + 4] += key * rhs.data[4];                              \
        mid[idx + 5] += key * rhs.data[5];                              \
        mid[idx + 6] += key * rhs.data[6];                              \
        mid[idx + 7] += key * rhs.data[7];                              \
    }
        DMUL_ACC(0);
        DMUL_ACC(1);
        DMUL_ACC(2);
        DMUL_ACC(3);
        DMUL_ACC(4);
        DMUL_ACC(5);
        DMUL_ACC(6);
        DMUL_ACC(7);
#undef DMUL_ACC
        retrieve(mid);
    }

    /* whether norm2 >= 4.0 */
    __device__ __forceinline__ bool escaping() const
    {
        return data[7] >= 4;
    }

private:
    __host__ __device__ constexpr FixedPoint8U30(
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

    /* finish carrying and write back the final result of multiplication */
    __device__ __forceinline__ void retrieve(uint64_t* mid)
    {
        mid[1] += mid[0] >> 30;
        mid[2] += mid[1] >> 30;
        mid[3] += mid[2] >> 30;
        mid[4] += mid[3] >> 30;
        mid[5] += mid[4] >> 30;
        mid[6] += mid[5] >> 30;
        mid[7] += (mid[6] >> 30) + (mid[6] & 0x3FFFFFFF >= 0x20000000);
        mid[8] += mid[7] >> 30;
        mid[9] += mid[8] >> 30;
        mid[10] += mid[9] >> 30;
        mid[11] += mid[10] >> 30;
        mid[12] += mid[11] >> 30;
        mid[13] += mid[12] >> 30;
        mid[14] += mid[13] >> 30;
        // narrow-conversion first, and then mask
        data[0] = static_cast<uint32_t>(mid[7]) & 0x3FFFFFFF;
        data[1] = static_cast<uint32_t>(mid[8]) & 0x3FFFFFFF;
        data[2] = static_cast<uint32_t>(mid[9]) & 0x3FFFFFFF;
        data[3] = static_cast<uint32_t>(mid[10]) & 0x3FFFFFFF;
        data[4] = static_cast<uint32_t>(mid[11]) & 0x3FFFFFFF;
        data[5] = static_cast<uint32_t>(mid[12]) & 0x3FFFFFFF;
        data[6] = static_cast<uint32_t>(mid[13]) & 0x3FFFFFFF;
        data[7] = static_cast<uint32_t>(mid[14]) & 0x3FFFFFFF;
    }

private:
    uint32_t data[8];
};