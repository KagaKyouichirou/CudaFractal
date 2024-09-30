#pragma once

#include <cuda_runtime.h>
#include <array>

/*
8 30-bit digits, each stored in uint32
*/

class FixedPoint8U30Naive
{
public:
    explicit FixedPoint8U30Naive();
    ~FixedPoint8U30Naive() = default;

    std::array<uint8_t, 30> intoBytes() const;
    static FixedPoint8U30Naive fromBytes(std::array<uint8_t, 30> const& bytes, bool negative);

    __host__ __device__ void flip();
    __host__ __device__ void add(FixedPoint8U30Naive const& other);
    __host__ __device__ void dou();
    __host__ __device__ void mul(FixedPoint8U30Naive& other);
    __host__ __device__ void mul(unsigned int f);
    __host__ __device__ void sqr();

    bool reasonable() const;
    bool zero() const;
    bool exceeds(FixedPoint8U30Naive const& bound);
    __host__ __device__ static bool checkNorm(FixedPoint8U30Naive const& x2, FixedPoint8U30Naive const& y2);

private:
    explicit constexpr FixedPoint8U30Naive(
        uint32_t d0,
        uint32_t d1,
        uint32_t d2,
        uint32_t d3,
        uint32_t d4,
        uint32_t d5,
        uint32_t d6,
        uint32_t d7
    );

private:
    uint32_t data[8];
};