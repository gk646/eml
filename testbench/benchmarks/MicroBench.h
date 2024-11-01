#ifndef EML_MICROBENCH_H
#define EML_MICROBENCH_H

using namespace eml;

#include <eml/config.h>

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

#define EML_BENCHMARK( type, func, iterations, size, ... )                                             \
    do {                                                                                                \
        /* Benchmark with time */                                                                       \
        const auto start = PlatformClock();                                                             \
        for (int32_t i = 0; i < iterations; ++i) {                                             \
            func(__VA_ARGS__);                                                                          \
        }                                                                                               \
        const auto end = PlatformClock();                                                               \
        const float elapsed = static_cast<float>(end - start) / 1'000'000.0F;                           \
        const float totalFLOP = iterations * powf(static_cast<float>(size), 3);                         \
                                                                                                        \
        /* Benchmark with cycles */                                                                     \
        const auto startCycle = PlatformCycleCount();                                                   \
        func(__VA_ARGS__);                                                                              \
        const auto endCycle = PlatformCycleCount();                                                     \
        const auto elapsedCycles = endCycle - startCycle;                                               \
        PlatformPrint("[Benchmark]:%s_%s\n\tCycles: %d cycles | Giga Ops: %.2f\n",                    \
                      __FUNCTION__,  GetTypeString<type>(), static_cast<long>(elapsedCycles),(totalFLOP / elapsed) / 1'000'000'000.0F);                  \
    } while (0)

#endif // EML_MICROBENCH_H