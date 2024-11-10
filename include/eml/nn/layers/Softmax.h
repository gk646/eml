#ifndef EML_LAYERS_SOFTMAX_H
#define EML_LAYERS_SOFTMAX_H

#include <eml/nn/Layer.h>

// ----------------------------------------------------------------
// Softmax
// ----------------------------------------------------------------
// ................................................................
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
// ................................................................

namespace eml::nn
{

struct Softmax final : Layer
{

    // Allocates a tensor of size (input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    Tensor<T> forward( const Tensor<T>& input );

    // Expects output a tensor of size (input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    void forward( const Tensor<T>& input, Tensor<T>& output );

  private:
    Pair padding;
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
Tensor<T> Softmax::forward( const Tensor<T>& input )
{
    Tensor<T> output{ input.h + padding.first * 2, input.w + padding.second * 2 };
    output.allocate();
    forward( input, output );
    return output;
}

template <typename T>
void Softmax::forward( const Tensor<T>& input, Tensor<T>& output )
{
    EML_ASSERT( output.h == input.h + padding.first * 2 && output.w == input.w + padding.second * 2, "Invalid shape" );
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