#ifndef EML_LAYERS_RELU_H
#define EML_LAYERS_RELU_H

// ================================================================
// ReLU
// ================================================================
// ................................................................
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
// ................................................................

namespace eml::nn
{

struct ReLU final
{

    // Returns an allocated tensor with the same shape as the input with the ReLU applied to all elements
    template <typename T>
    [[nodiscard]] static Tensor<T> forward( const Tensor<T>& input );

    // Applies the ReLU inplace
    template <typename T>
    static void forwardI( Tensor<T>& input );
};

} // namespace eml::nn

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

namespace eml::nn
{

template <typename T>
Tensor<T> ReLU::forward( const Tensor<T>& input )
{
    Tensor<T> output = input.copyTensor();
    forwardI(  output );
    return output;
}

template <typename T>
void ReLU::forwardI( Tensor<T>& input )
{
    constexpr auto simdSize = static_cast<int32_t>( xsimd::batch<T>::size );

    int32_t i = 0;
    if constexpr( simdSize > 0 )
    {
        xsimd::batch zero_batch = xsimd::broadcast<T>( 0 );
        const int32_t vec_size = input.size - input.size % simdSize;
        for( ; i < vec_size; i += simdSize )
        {
            const auto loaded = xsimd::load_unaligned( &input[ i ] );
            const auto comp = xsimd::max( zero_batch, loaded );
            comp.store_unaligned( &input[ i ] );
        }
    }

    for( ; i < input.size; ++i )
    {
        if( input[ i ] < 0 )
        {
            input[ i ] = 0;
        }
    }
}

} // namespace eml::nn

#endif // EML_LAYERS_RELU_H