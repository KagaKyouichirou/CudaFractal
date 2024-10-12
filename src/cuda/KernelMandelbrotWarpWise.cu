#include "FixedPoint8U30.h"
#include "TaskArgs.h"

using u32 = unsigned int;
using u64 = unsigned long long;

__global__ void kernelMandelbrotWarpWise(cudaSurfaceObject_t surf, u32 const* data, u32 limit);

extern "C" {
    void launchKernelMandelbrotWarpWise(cudaSurfaceObject_t surf, void const* pTaskArgs)
    {
        auto task = static_cast<TaskArgs const*>(pTaskArgs);
        FixedPoint8U30 oX = task->h;
        oX.mul(task->dGrid.x * task->dBlock.x - 1);
        oX.flip();
        oX.add(task->x);
        FixedPoint8U30 oY = task->h;
        oY.mul(task->dGrid.y * task->dBlock.y - 1);
        oY.flip();
        oY.add(task->y);
        FixedPoint8U30 step = task->h;
        step.mul(2);

        u32* data = nullptr;
        cudaMalloc(&data, 24 * sizeof(u32));

        cudaMemcpy(&data[0], oX.raw(), 8 * sizeof(u32), cudaMemcpyHostToDevice);
        cudaMemcpy(&data[16], oY.raw(), 8 * sizeof(u32), cudaMemcpyHostToDevice);
        cudaMemcpy(&data[8], step.raw(), 8 * sizeof(u32), cudaMemcpyHostToDevice);

        dim3 dBlock = task->dBlock;
        dBlock.x <<= 5;
        kernelMandelbrotWarpWise<<<task->dGrid, dBlock>>>(surf, data, task->limit);

        cudaFree(data);
        data = nullptr;
    }
}

/*
    data:                 <-----oY-----><----step----><-----oX----->

    warp:    31        24  23        16  15        08  07        00
    e:                    <-----eY-----><----norm2---><-----eX----->
    u:      <----imgn2---><----imgn----><----real2---><----real---->
    v:      <------------vi------------><------------vr------------>
*/
__global__ void kernelMandelbrotWarpWise(cudaSurfaceObject_t surf, u32 const* data, u32 limit)
{
    u32 const lane = threadIdx.x & 0x1F;
    u32 const m_lane = 1u << lane;

    u32 const sub_lane = lane & 15;

    // [x, y]
    u32 const coord[2]{(blockIdx.x * blockDim.x + threadIdx.x) >> 5, blockIdx.y * blockDim.y + threadIdx.y};

    u64 e = 0;

    constexpr u32 m_lower = 0x00FF00FF;
    constexpr u32 m_upper = 0xFF00FF00;

    // eX = step.mul(x).add(oX); eY = step.mul(y).add(oY);
    // keep eX and eY in signed form
    if (m_lane & m_lower) {
        e = data[8 + (lane & 7)];

        e *= coord[lane > 15];
        u32 o[2];
        o[0] = data[lane];
        bool o_sign;
        if (7 == sub_lane) {
            o_sign = o[0] & 0x80000000;
            o[0] &= 0x3FFFFFFF;
        }
        o_sign = __shfl_sync(m_lower, o_sign, 7, 16);
        o[1] = 0x3FFFFFFF - o[0] + (0 == sub_lane);
        e += o[o_sign];
    }  // if (m_lane & m_lower)

/* from idx to (idx + 1) */
#define CARRY(var, idx)                                             \
    {                                                               \
        u64 const c = __shfl_up_sync(0xFFFFFFFF, var, 1, 16) >> 30; \
        if (idx + 1 == sub_lane) {                                  \
            var += c;                                               \
        }                                                           \
    }

    CARRY(e, 0);
    CARRY(e, 1);
    CARRY(e, 2);
    CARRY(e, 3);
    CARRY(e, 4);
    CARRY(e, 5);
    CARRY(e, 6);
    e &= 0x3FFFFFFF;

    u64 u = 0;
    bool u_sign = false;
    u32 k = 0;
    // loop
    while (k < limit) {
        // real.add(eX); imgn.add(eY);
        // finish carrying on real and imgn
        // and turn them into unsigned form
        {
#define CARRY_U_LOWER()  \
    {                    \
        CARRY(u, 0);     \
        CARRY(u, 1);     \
        CARRY(u, 2);     \
        CARRY(u, 3);     \
        CARRY(u, 4);     \
        CARRY(u, 5);     \
        CARRY(u, 6);     \
        u &= 0x3FFFFFFF; \
    }
            if (m_lane & m_lower) {
                u += e;
            }
            CARRY_U_LOWER();
            if (m_lane & m_lower) {
                if (7 == sub_lane) {
                    u_sign = u >= 0x20000000;
                }
                u_sign = __shfl_sync(m_lower, u_sign, 7, 16);
                if (u_sign) {
                    u = 0x3FFFFFFF - u;
                    u += 0 == sub_lane;
                }
            }  // if (m_lane & m_lower)

            if (__any_sync(0xFFFFFFFF, u_sign)) {
                CARRY_U_LOWER();
            }
#undef CARRY_U_LOWER
        }

        // real2 = real * real; imgn2 = imgn * imgn;
        {
            u64 v = 0;
#define SQR_ACC(idx)                                 \
    {                                                \
        u64 t = __shfl_sync(0xFFFFFFFF, u, idx, 16); \
        t *= __shfl_up_sync(0xFFFFFFFF, u, idx, 16); \
        if (m_lane & (m_lower << idx)) {             \
            v += t;                                  \
        }                                            \
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

            CARRY(v, 0);
            CARRY(v, 1);
            CARRY(v, 2);
            CARRY(v, 3);
            CARRY(v, 4);
            CARRY(v, 5);
            CARRY(v, 6);
            // rounding
            u64 t = __shfl_sync(0xFFFFFFFF, v, 6, 16);
            v += (7 == sub_lane) && ((t & 0x3FFFFFFF) >= 0x20000000);
            CARRY(v, 7);
            CARRY(v, 8);
            CARRY(v, 9);
            CARRY(v, 10);
            CARRY(v, 11);
            CARRY(v, 12);
            CARRY(v, 13);
            v &= 0x3FFFFFFF;
            t = __shfl_up_sync(0xFFFFFFFF, v, 1, 16);
            if (m_lane & m_upper) {
                u = t;
            }
        }

        // norm2 = real2 + imgn2
        {
            u64 t = __shfl_down_sync(0xFFFFFFFF, u, 16, 32);
            if (m_lane & 0x0000FF00) {
                e = t + u;
            }
            CARRY(e, 8);
            CARRY(e, 9);
            CARRY(e, 10);
            CARRY(e, 11);
            CARRY(e, 12);
            CARRY(e, 13);
            CARRY(e, 14);
            e &= 0x3FFFFFFF;
        }

        // check if norm2 < 4.0
        {
            u32 flag;
            if (15 == lane) {
                flag = e >= 4;
            }
            flag = __shfl_sync(0xFFFFFFFF, flag, 15, 32);
            if (flag) {
                break;
            }
        }

        // vi = imgn * real * 2
        // write back to imgn in signed form
        // but no carrying afterwrads
        {
            // determine sign for imgn
            bool const real_sign = __shfl_up_sync(0xFFFFFFFF, u_sign, 16, 32);
            constexpr u32 m_imgn = 0x00FF0000;
            if (m_lane & m_imgn) {
                u_sign ^= real_sign;
            }
            u64 v = 0;
#define MUL_ACC(idx)                                                                                            \
    {                                                                                                           \
        u64 const t = 2llu * __shfl_sync(0xFFFFFFFF, u, idx, 16) * __shfl_up_sync(0xFFFFFFFF, u, idx + 16, 32); \
        if (m_lane & (m_imgn << idx)) {                                                                         \
            v += t;                                                                                             \
        }                                                                                                       \
    }
            MUL_ACC(0);
            MUL_ACC(1);
            MUL_ACC(2);
            MUL_ACC(3);
            MUL_ACC(4);
            MUL_ACC(5);
            MUL_ACC(6);
            MUL_ACC(7);
#undef MUL_ACC

            CARRY(v, 0);
            CARRY(v, 1);
            CARRY(v, 2);
            CARRY(v, 3);
            CARRY(v, 4);
            CARRY(v, 5);
            CARRY(v, 6);
            // rounding
            u64 t = __shfl_sync(0xFFFFFFFF, v, 22, 32);
            v += (23 == lane) && ((t & 0x3FFFFFFF) >= 0x20000000);
            CARRY(v, 7);
            CARRY(v, 8);
            CARRY(v, 9);
            CARRY(v, 10);
            CARRY(v, 11);
            CARRY(v, 12);
            CARRY(v, 13);
            v &= 0x3FFFFFFF;
            t = __shfl_down_sync(0xFFFFFFFF, v, 7, 32);
            if (m_lane & m_imgn) {
                u = t;
                if (u_sign) {
                    u = 0x3FFFFFFF - u;
                    u += 16 == lane;
                }
            }
        }

        // real = imgn2.complement().add(real2)
        // keep in signed form
        // no carrying
        {
            constexpr u32 m_real = 0x000000FF;
            u64 t = __shfl_down_sync(0xFFFFFFFF, u, 24, 32);
            if (m_lane & m_real) {
                u = 0x3FFFFFFF - t;
                u += 0 == lane;
            }
            t = __shfl_down_sync(0xFFFFFFFF, u, 8, 32);
            if (m_lane & m_real) {
                u += t;
            }
        }

        k++;
    }  // while (k < limit)

#undef CARRY

    if (0 == lane) {
        surf2Dwrite(static_cast<float>(k) / limit, surf, coord[0] * sizeof(float), coord[1], cudaBoundaryModeTrap);
    }
}
