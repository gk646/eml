#ifndef EML_MATHUTIL_H
#define EML_MATHUTIL_H

#include <random>

// ================================================================
// Math
// ================================================================
// ................................................................
// ................................................................
// ................................................................


namespace eml
{



// Returns a random float between min and max (inclusive)
float GetRandomFloat( float min, float max );

// Returns the amount of digits needed to represent the number
template <typename T>
int32_t GetDigitCount( T num );

template <typename T>
T Clamp( T val, T min, T max );

template <typename T>
T Exp( const T& val );

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

template <typename T>
T Clamp( const T val, const T min, const T max )
{
    if( val < min )
        return min;
    if( val > max )
        return max;
    return val;
}

template <typename T>
T Exp( const T& val )
{
}

} // namespace eml

#endif // EML_MATHUTIL_H