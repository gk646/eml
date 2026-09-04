#ifndef EML_MACROS_H
#define EML_MACROS_H

#define EML_CHECK_TYPE_SUPPORTED()                                                                                     \
    static_assert( std::is_same_v<T, float> || std::is_same_v<T, bool> || std::is_same_v<T, qint8_t> ||                \
                   std::is_same_v<T, qint16_t> || std::is_same_v<T, qint32_t> && "Unsupported type specified" )

#define EML_FRIEND_LAYERS()                                                                                            \
    friend struct Linear<T>;                                                                                           \
    friend struct Conv1D<T>;                                                                                           \
    friend struct Conv2D<T>;                                                                                           \
    friend struct ReLU<T>;

#define EML_MODEL_ALLOCATE( tensor )                                                                                   \


#if defined( __GNUC__ ) || defined( __clang__ )
#define restrict __restrict__
#else
#define restrict
#endif

#endif // EML_MACROS_H