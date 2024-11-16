#ifndef EML_MICROBENCH_H
#define EML_MICROBENCH_H

#include <eml/config.h>
#include <type_traits>

using namespace eml;

namespace eml
{

struct BenchmarkData final
{
};

template <typename T>
constexpr const char* GetTypeString()
{
    if constexpr( std::is_same_v<T, qint8_t> )
    {
        return "qint8";
    }
    else if constexpr( std::is_same_v<T, qint16_t> )
    {
        return "qint16";
    }
    else if constexpr( std::is_same_v<T, qint32_t> )
    {
        return "qint32";
    }
    else if constexpr( std::is_same_v<T, float> )
    {
        return "float";
    }
    else
    {
        return "unknown";
    }
}

} // namespace eml

#define EML_BENCHMARK_FLOPS( type, func, iterations, ops, ... )                                                        \
    const auto start = PlatformClock();                                                                                \
    for( int32_t i = 0; i < iterations; ++i )                                                                          \
    {                                                                                                                  \
        func( __VA_ARGS__ );                                                                                           \
    }                                                                                                                  \
    const auto end = PlatformClock();                                                                                  \
    const float elapsed = static_cast<float>( end - start ) / 1'000'000.0F;                                            \
    const float totalOps = iterations * ops;                                                                           \
                                                                                                                       \
    const auto startCycle = PlatformCycleCount();                                                                      \
    for( int32_t i = 0; i < iterations; ++i )                                                                          \
    {                                                                                                                  \
        func( __VA_ARGS__ );                                                                                           \
    }                                                                                                                  \
    const auto endCycle = PlatformCycleCount();                                                                        \
    const auto elapsedCycles = endCycle - startCycle;                                                                  \
                                                                                                                       \
    if constexpr( std::is_floating_point<type>::value )                                                                \
    {                                                                                                                  \
        PlatformPrint( "[Benchmark]: %s:%s\n\t==> %4.1f ms : %4.1f (mults/cycle) : %4.1f (GFLOPS) \n", __FUNCTION__,   \
                       GetTypeString<type>(), elapsed * 1000.0F, (float)totalOps / (float)elapsedCycles,               \
                       ( totalOps / elapsed ) / 1'000'000'000.0F );                                                    \
    }                                                                                                                  \
    else                                                                                                               \
    {                                                                                                                  \
        PlatformPrint( "Benchmark]: %s:%s\n\t==> %4.1f ms : %4.1f (mults/cycle) : %4.1f (GIOPS) \n", __FUNCTION__,     \
                       GetTypeString<type>(), elapsed * 1000.0F, (float)totalOps / (float)elapsedCycles,               \
                       ( totalOps / elapsed ) / 1'000'000'000.0F );                                                    \
    }

#endif // EML_MICROBENCH_H