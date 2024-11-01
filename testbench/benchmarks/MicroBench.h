#ifndef EML_MICROBENCH_H
#define EML_MICROBENCH_H

using namespace eml;

template <typename T>
constexpr const char* GetTypeString(T) {
    if constexpr (std::is_same_v<T, qint8>) {
        return "qint8";
    } else if constexpr (std::is_same_v<T, qint16>) {
        return "qint16";
    } else if constexpr (std::is_same_v<T, qint32>) {
        return "qint32";
    } else if constexpr (std::is_same_v<T, float>) {
        return "float";
    } else if constexpr (std::is_same_v<T, int>) {
        return "int";
    } else {
        return "unknown";
    }
}

#define EML_BENCH_OPS_IT( type, func, iterations, size, ... )                                                          \
    const auto start = PlatformClock();                                                                                \
    for( volatile int32_t i = 0; i < count; ++i )                                                                      \
    {                                                                                                                  \
        func( __VA_ARGS__ );                                                                                           \
    }                                                                                                                  \
    const auto end = PlatformClock();                                                                                  \
    const float elapsed = static_cast<float>( end - start ) / 1'000'000.0F;                                            \
    const float totalFLOP = count * powf( static_cast<float>( size ), 3 );                                             \
    PlatformPrint( "%s%s%4d: %5.0f GFLOPS\n", #type, __FUNCTION__, size, ( totalFLOP / elapsed ) / 1'000'000'000.0F );

#endif // EML_MICROBENCH_H