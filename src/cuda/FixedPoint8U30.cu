#include "FixedPoint8U30.h"

std::array<uint8_t, 30> FixedPoint8U30::intoBytes() const
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

FixedPoint8U30 FixedPoint8U30::fromBytes(std::array<uint8_t, 30> const& bytes)
{
    return FixedPoint8U30(
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
        // clang-format on
    );
}

uint32_t const* FixedPoint8U30::raw() const
{
    return data;
}

bool FixedPoint8U30::reasonable() const
{
    return data[7] < 4;
}

bool FixedPoint8U30::nonzero() const
{
    return data[0] || data[1] || data[2] || data[3] || data[4] || data[5] || data[6] || data[7];
}

bool FixedPoint8U30::exceeds(FixedPoint8U30 const& bound) const
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
