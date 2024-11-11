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

    // Allocates a tensor of size (input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    static Tensor<T> forward( const Tensor<T>& input );

    // Expects output a tensor of size (input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    static void forward( Tensor<T>& input );

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
static Tensor<T> ReLU::forward( const Tensor<T>& input )
{
    Tensor<T> output{ };
    output.allocate();
    forward( output );
    return output;
}

template <typename T>
static void ReLU::forward( const Tensor<T>& input )
{

    int32_t directionH = padding.first > 0 ? -1 : 1;
    int32_t indexH = padding.first;
    for( int32_t h = 0; h < output.h; h++ )
    {
        int32_t directionW = padding.second > 0 ? -1 : 1;
        int32_t indexW = padding.second;
        for( int32_t w = 0; w < output.w; w++ )
        {
            output[ h * output.w + w ] = input[ indexH * input.w + indexW ];
            indexW += directionW;
            if( indexW == 0 )
                directionW = 1;
            else if( w == input.w + padding.second - 2 )
                directionW = -1;
        }
        indexH += directionH;
        if( indexH == 0 )
            directionH = 1;
        else if( h == input.h + padding.first - 2 )
            directionH = -1;
    }
}

} // namespace eml::nn

#endif // EML_LAYERS_SOFTMAX_H