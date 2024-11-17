#ifndef EML_MACROS_H
#define EML_MACROS_H

#define EML_CHECK_TYPE_SUPPORTED()                                                                                     \
    static_assert( std::is_same_v<T, float> || std::is_same_v<T, qint8_t> || std::is_same_v<T, qint16_t> ||            \
                   std::is_same_v<T, qint32_t> && "Unsupported type specified" )

#define EML_FRIEND_LAYERS()                                                                                            \
    friend struct Linear<T>;                                                                                           \
    friend struct Conv1D<T>;                                                                                           \
    friend struct Conv2D<T>;                                                                                           \
    friend struct ReLU<T>;

#define EML_MODEL_ALLOCATE( tensor )                                                                                   \
    auto* tensor##Memory = model->context.requestConsistentMemory( tensor.size, sizeof( T ) );                                   \
    tensor.allocateCustom( tensor##Memory, tensor.size );                                                              \
    if( model->withTraining )                                                                                          \
    {                                                                                                                  \
        tensor.grad = (Tensor<T>*)model->context.requestConsistentMemory( 1, sizeof( Tensor<T> ) );                              \
        *tensor.grad = Tensor<T>{ model->batches, tensor.c, tensor.h, tensor.w };                                      \
        auto* gradMemory = model->context.requestConsistentMemory( tensor.grad->size, sizeof( T ) );                             \
        tensor.grad->allocateCustom( gradMemory, tensor.grad->size );                                                  \
    }

#if defined( __GNUC__ ) || defined( __clang__ )
#define restrict __restrict__
#else
#define restrict
#endif

#endif // EML_MACROS_H