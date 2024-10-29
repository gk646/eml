#ifndef EML_MATHUTIL_H
#define EML_MATHUTIL_H

#include <random>

namespace eml
{

// Returns a random float between min and max (inclusive)
float GetRandomFloat( float min, float max )
{
    static std::mt19937 engine{ std::random_device{}() };
    std::uniform_real_distribution<float> distribution( min, max );
    return distribution( engine );
}

} // namespace eml
#endif // EML_MATHUTIL_H