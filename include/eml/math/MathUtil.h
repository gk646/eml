#ifndef EML_MATHUTIL_H
#define EML_MATHUTIL_H

#include <random>

namespace eml
{
enum class PaddingMode
{
    ZEROS,
    REFLECT,
    REPLICATE,
    CIRCULAR
};

struct Pair
{
    int first;
    int second;
};

// Returns a random float between min and max (inclusive)
float GetRandomFloat( float min, float max );

// Returns the amount of digits needed to represent the number
template <typename T>
int32_t GetDigitCount( T num );
} // namespace eml

// IMPLEMENTATION
//
//
//
//
//
//
//
//
//
//
//
//
//
//
//

namespace eml
{

inline float GetRandomFloat( float min, float max )
{

    static std::mt19937 engine{ std::random_device{}() };
    std::uniform_real_distribution<float> distribution( min, max );
    return distribution( engine );
}

template <typename T>
int32_t GetDigitCount( T num )
{
    return 1;
}

} // namespace eml

#endif // EML_MATHUTIL_H