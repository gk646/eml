#ifndef EML_LAYERS_LEAKYRELU_H
#define EML_LAYERS_LEAKYRELU_H

// ================================================================
// LeakyReLU
// ================================================================
// ................................................................
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.LeakyReLU.html
// ................................................................

namespace eml::nn
{

struct LeakyReLU final
{

    // Applies ReLU as normal but instead of zero multiplies negative values with the given slop
    explicit LeakyReLU( float negativeSlope = 0.01 );

    // Returns an allocated tensor with the same shape as the input with the ReLU applied to all elements
    template <typename T>
    [[nodiscard( "Allocates output tensor" )]] Tensor<T> forward( const Tensor<T>& input );

    // Applies the ReLU inplace
    template <typename T>
    void forward( Tensor<T>&& input );

    float negativeSlope;
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

inline LeakyReLU::LeakyReLU( const float negativeSlope ) : negativeSlope( negativeSlope )
{
}

template <typename T>
Tensor<T> LeakyReLU::forward( const Tensor<T>& input )
{
    Tensor<T> output = input.copyTensor();
    forward( std::move( output ) );
    return output;
}

template <typename T>
void LeakyReLU::forward( Tensor<T>&& input )
{
    constexpr auto simdSize = static_cast<int32_t>( xsimd::batch<T>::size );

    int32_t i = 0;
    if constexpr( simdSize > 0 )
    {
        xsimd::batch zero_batch = xsimd::broadcast<T>( 0 );
        xsimd::batch slope = xsimd::broadcast<T>( negativeSlope );

        const int32_t vec_size = input.size - input.size % simdSize;
        for( ; i < vec_size; i += simdSize )
        {
            const auto loaded = xsimd::load_unaligned( &input[ i ] );
            const auto mask = loaded < zero_batch;
            const auto comp = xsimd::select( mask, loaded * slope, loaded );
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

#endif // EML_LAYERS_LEAKYRELU_H