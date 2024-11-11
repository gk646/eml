#ifndef EML_UTIL_TYPES_H
#define EML_UTIL_TYPES_H

#include <cstdint>

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

    void print( const char* name = nullptr ) const;
};

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

inline void Tuple::print( const char* name ) const
{
    PlatformPrint( "%s: [%3d, %3d, %3d, %3d]", name ? name : "Tuple", first, second, third, fourth );
}

} // namespace eml

#endif // EML_UTIL_TYPES_H