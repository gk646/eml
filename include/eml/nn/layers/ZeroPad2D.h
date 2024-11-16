#ifndef EML_NN_LAYERS_ZEROPAD2D_H
#define EML_NN_LAYERS_ZEROPAD2D_H

// ================================================================
// ReflectionPad2D
// ================================================================
// ................................................................
// ................................................................
// Doc: https://pytorch.org/docs/stable/generated/torch.nn.ZeroPad2d.html
// ................................................................

namespace eml::nn
{

struct ZeroPad2D final
{

    explicit ZeroPad2D( Pair padding );

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

inline ZeroPad2D::ZeroPad2D( const Pair padding ) : padding( padding )
{
}

template <typename T>
Tensor<T> ZeroPad2D::forward( const Tensor<T>& input )
{
    Tensor<T> output{ input.n, input.c, input.h + padding.first * 2, input.w + padding.second * 2 };
    output.allocate();
    forwardI( input, output );
    return output;
}

template <typename T>
void ZeroPad2D::forwardI( const Tensor<T>& input, Tensor<T>& output )
{
    EML_ASSERT( output.h == input.h + padding.first * 2 && output.w == input.w + padding.second * 2, "Invalid shape" );

    int32_t chOffOut = 0;
    int32_t chOffIn = 0;
    for( int32_t n = 0; n < input.n; ++n )
    {
        for( int32_t c = 0; c < output.c; ++c )
        {
            int32_t matrixOffOut = chOffOut;
            int32_t matrixOffIn = chOffIn;
            for( int32_t h = 0; h < output.h; h++ )
            {
                if( h < padding.first || h >= input.h + padding.first )
                {
                    for( int32_t w = 0; w < output.w; w++ )
                    {
                        output[ matrixOffOut + w ] = 0;
                    }
                }
                else
                {
                    for( int32_t w = 0; w < output.w; w++ )
                    {
                        if( w >= padding.second && w < input.w + padding.second )
                        {
                            output[ matrixOffOut + w ] = input[ matrixOffIn + w - padding.second ];
                        }
                        else
                        {
                            output[ matrixOffOut + w ] = 0;
                        }
                    }
                    matrixOffIn += input.w;
                }
                matrixOffOut += output.w;
            }
            chOffOut += output.hw;
            chOffIn += input.hw; // Automatically advances to next batch
        }
    }
}

} // namespace eml::nn

#endif // EML_NN_LAYERS_ZEROPAD2D_H