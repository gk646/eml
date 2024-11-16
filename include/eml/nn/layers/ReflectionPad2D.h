#ifndef EML_NN_LAYERS_REFLECTIONPAD2D_H
#define EML_NN_LAYERS_REFLECTIONPAD2D_H

// ================================================================
// ReflectionPad2D
// ================================================================
// ................................................................
// Applies padding to each matrix within the given tensor
//      - Iterates every channel in every batch and pads every matrix
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.ReflectionPad2d.html
// ................................................................

namespace eml::nn
{

struct ReflectionPad2D final
{
    explicit ReflectionPad2D( Pair padding );

    // ============ Forward ============

    // Allocates a tensor of size (*, *, input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    Tensor<T> forward( const Tensor<T>& input );

    // Expects output a tensor of size (*, *, input.h + padding.first * 2, input.w + padding.second * 2)
    template <typename T>
    void forwardI( const Tensor<T>& input, Tensor<T>& output );

    // ============ Variables ============

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

inline ReflectionPad2D::ReflectionPad2D( const Pair padding ) : padding( padding )
{
}

template <typename T>
Tensor<T> ReflectionPad2D::forward( const Tensor<T>& input )
{
    Tensor<T> output{ input.n, input.c, input.h + padding.first * 2, input.w + padding.second * 2 };
    output.allocate();
    forwardI( input, output );
    return output;
}

template <typename T>
void ReflectionPad2D::forwardI( const Tensor<T>& input, Tensor<T>& output )
{
    EML_ASSERT( output.h == input.h + padding.first * 2 && output.w == input.w + padding.second * 2, "Invalid shape" );

    int32_t chOffIn = 0;
    int32_t chOffOut = 0;
    for( int32_t n = 0; n < input.n; ++n )
    {
        for( int32_t c = 0; c < input.c; ++c )
        {
            int32_t directionH = padding.first > 0 ? -1 : 1;
            int32_t indexH = padding.first;
            for( int32_t h = 0; h < output.h; h++ )
            {
                int32_t directionW = padding.second > 0 ? -1 : 1;
                int32_t indexW = padding.second;
                for( int32_t w = 0; w < output.w; w++ )
                {
                    output[ chOffOut + h * output.w + w ] = input[ chOffIn + indexH * input.w + indexW ];
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
            chOffIn += input.hw;
            chOffOut += output.hw; // Automatically advances to new batch
        }
    }
}

} // namespace eml::nn

#endif // EML_NN_LAYERS_REFLECTIONPAD2D_H