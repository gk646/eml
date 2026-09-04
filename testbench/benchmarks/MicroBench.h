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
    do                                                                                                                 \
    {                                                                                                                  \
        const auto startClock = PlatformClock();                                                                       \
        const auto startCycle = PlatformCycleCount();                                                                  \
        for( int32_t i = 0; i < iterations; ++i )                                                                      \
        {                                                                                                              \
            func( __VA_ARGS__ );                                                                                       \
        }                                                                                                              \
        const auto endCycle = PlatformCycleCount();                                                                    \
        const auto endClock = PlatformClock();                                                                         \
                                                                                                                       \
        const auto elapsedCycles = endCycle > startCycle ? endCycle - startCycle : 0;                                  \
        const auto elapsedClock = endClock > startClock ? endClock - startClock : 0;                                   \
                                                                                                                       \
        const double elapsedMs = static_cast<double>( elapsedClock ) / 1'000.0;                                        \
        const double totalOps = static_cast<double>( iterations ) * ops;                                               \
        const double multsPerCycle = elapsedCycles > 0 ? totalOps / static_cast<double>( elapsedCycles ) : 0.0;        \
        const double gflops = elapsedMs > 0 ? ( totalOps / elapsedMs ) / 1'000'000.0 : 0.0;                            \
                                                                                                                       \
        if constexpr( std::is_floating_point<type>::value )                                                            \
        {                                                                                                              \
            PlatformPrint( "[Benchmark]: %s:%s\n\t==> %4.1f ms : %4.1f (mults/cycle) : %4.1f (GFLOPS) \n", __func__,   \
                           GetTypeString<type>(), elapsedMs, multsPerCycle, gflops );                                  \
        }                                                                                                              \
        else                                                                                                           \
        {                                                                                                              \
            PlatformPrint( "[Benchmark]: %s:%s\n\t==> %4.1f ms : %4.1f (mults/cycle) : %4.1f (GIOPS) \n", __func__,    \
                           GetTypeString<type>(), elapsedMs, multsPerCycle, gflops );                                  \
        }                                                                                                              \
    } while( 0 )

#endif // EML_MICROBENCH_H