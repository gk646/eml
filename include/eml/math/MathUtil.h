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
    int32_t first;
    int32_t second;
};

struct Tuple
{
    int32_t first;
    int32_t second;
    int32_t third;
    int32_t fourth;

    bool operator==( const Tuple& rhs ) const;
    bool operator!=( const Tuple& rhs ) const;
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

inline bool Tuple::operator==( const Tuple& rhs ) const
{
    return first == rhs.first && second == rhs.second && third == rhs.third && fourth == rhs.fourth;
}

inline bool Tuple::operator!=( const Tuple& rhs ) const
{
    return !( *this == rhs );
}

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