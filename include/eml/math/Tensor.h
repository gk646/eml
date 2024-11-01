#ifndef EML_TENSOR_H
#define EML_TENSOR_H

#include <type_traits>

#include <eml/config.h>
#include <eml/math/MathUtil.h>

// ----------------------------------------------------------------
// Tensor
// ----------------------------------------------------------------
// ................................................................
//  4 dimensional tensor
// ................................................................
// Misc:
//
// ................................................................

namespace eml
{
template <typename T>
struct Tensor final
{
    // Constructs a new tensor with dims
    //      - N: Batch size
    //      - C: Channels
    //      - H: Vertical size
    //      - W: Horizontal size
    Tensor( int32_t N, int32_t C, int32_t H, int32_t W );

    // Constructs a new image tensor with dims
    //      - C: Channels
    //      - H: Vertical size
    //      - W: Horizontal size
    Tensor( int32_t C, int32_t H, int32_t W );

    // Constructs a new matrix tensor with dims
    //      - H: Vertical size
    //      - W: Horizontal size
    Tensor( int32_t H, int32_t W );

    // Constructs a new vector tensor with dims
    //      - W: Horizontal size
    explicit Tensor( int32_t W );

    // Access via flattened array index
    T& operator[]( int32_t idx );
    T operator[]( int32_t idx ) const;

    // Allocates memory according to the current size ONLY when data is nullptr - uses PlatformAlloc
    void allocate();

    // Frees the memory and sets data to nullptr
    void free();

    // Initializes the tensor with user managed memory
    void allocateCustom( void* memory, int count );

    // Sets the data to nullptr and returns the memory
    void* freeCustom();

    // Prints the tensor with PlatformPrint()
    void print( const char* name = "Tensor" );

    // Returns true if the tensor is allocated
    [[nodiscard]] bool isAllocated() const;

    // Returns true if the tensor is allocated custom
    [[nodiscard]] bool isAllocatedCustom() const;

    // ------------ Data ------------

    T* data = nullptr; // Data pointer
    int32_t capacity = 0; // Allocated size

    int32_t size = 0; // Maximum elements
    int32_t n = 0; // Batch
    int32_t c = 0; // Channels
    int32_t h = 0; // Height / Rows
    int32_t w = 0; // Width / Columns

    int32_t hw = 0; // Channel stride
    int32_t chw = 0; // Batch stride

    // ------------ Metadata ------------

    int32_t scale = 0; // Scale for quantized values
    bool customAllocated = false;
};

} // namespace eml

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

namespace eml
{

template <typename T>
Tensor<T>::Tensor( const int32_t N, const int32_t C, const int32_t H, const int32_t W )
    : size( N * C * H * W ), n( N ), c( C ), h( H ), w( W ), hw( H * W ), chw( C * H * W )
{
    EML_ASSERT( N > 0 && C > 0 && H > 0 && W > 0, "Dimensions must be at least 1" );
#ifdef EML_DEBUG
    const uint64_t size = n * c * h * w;
    EML_ASSERT( size < INT32_MAX, "Size overflows" );
    const uint64_t stride = h * w;
    EML_ASSERT( stride < INT32_MAX, "Stride overflows" );
#endif
}

template <typename T>
Tensor<T>::Tensor( int32_t C, int32_t H, int32_t W ) : Tensor( 1, C, H, W )
{
}

template <typename T>
Tensor<T>::Tensor( int32_t H, int32_t W ) : Tensor( 1, 1, H, W )
{
}
template <typename T>
Tensor<T>::Tensor( int32_t W ) : Tensor( 1, 1, 1, W )
{
}

template <typename T>
T& Tensor<T>::operator[]( int32_t idx )
{
    EML_ASSERT( idx < capacity, "Out of bounds access" );
    EML_ASSERT( idx < size, "Out of bounds access" );
    return data[ idx ];
}

template <typename T>
T Tensor<T>::operator[]( int32_t idx ) const
{
    EML_ASSERT( idx < capacity, "Out of bounds access" );
    EML_ASSERT( idx < size, "Out of bounds access" );
    return data[ idx ];
}

template <typename T>
void Tensor<T>::allocate()
{
    EML_ASSERT( capacity == 0, "Calling allocate() on an already allocated tensor" );
    EML_ASSERT( size > 0, "Calling allocate() on an empty tensor - likely a mistake" );
    data = static_cast<T*>( PlatformAlloc( sizeof( T ) * size ) );
    capacity = size;
}

template <typename T>
void Tensor<T>::free()
{
    EML_ASSERT( !customAllocated, "Calling free() on custom allocated tensor! Call freeCustom()" );
    EML_ASSERT( capacity > 0, "Calling free() on empty an tensor - likely a mistake" );
    PlatformFree( data );
    data = nullptr;
    capacity = 0;
}

template <typename T>
void Tensor<T>::allocateCustom( void* memory, const int count )
{
    EML_ASSERT( static_cast<T*>( memory ) != nullptr && count > 0, "Calling stackAllocate() with invalid parameters" );
    EML_ASSERT( data == nullptr, "Cannot initialize an allocated tensor! Call free() or freeCustom() first" );
    EML_ASSERT( count >= size, "Passed memory is too small to hold the tensor data!" );
    data = static_cast<T*>( memory );
    capacity = count;
    customAllocated = true;
}

template <typename T>
void* Tensor<T>::freeCustom()
{
    EML_ASSERT( customAllocated, "Calling freeCustom() on non-custom allocated tensor! Call free()" );
    EML_ASSERT( capacity > 0, "Calling free() on an empty tensor - likely a mistake" );
    const auto* temp = data;
    data = nullptr;
    capacity = 0;
    return temp;
}

template <typename T>
void Tensor<T>::print( const char* name )
{

    const char* format = std::is_floating_point_v<T> ? "%4.1f " : "%d ";
    int32_t batchIndex = 0;
    PlatformPrint( "%s:\n", name );
    for( int32_t batch = 0; batch < n; ++batch )
    {
        PlatformPrint( "[" );
        for( int32_t row = 0; row < h; ++row )
        {
            int32_t rowIndex = batchIndex + row * w;
            for( int32_t channel = 0; channel < c; ++channel )
            {
                if( row == 0 )
                    PlatformPrint( "[" );
                else
                    PlatformPrint( "|" );
                for( int32_t col = 0; col < w; ++col )
                {
                    PlatformPrint( format, this->operator[]( rowIndex + col ) );
                }
                if( row == h - 1 )
                    PlatformPrint( "]" );
                else
                    PlatformPrint( "|" );
                rowIndex += hw;
            }
            if( row == h - 1 )
                PlatformPrint( "]\n" );
            else
            PlatformPrint( "\n " );
        }
        batchIndex += chw;
    }
    PlatformPrint( "\n" );
}

template <typename T>
bool Tensor<T>::isAllocated() const
{
    return data != nullptr;
}

template <typename T>
bool Tensor<T>::isAllocatedCustom() const
{
    return isAllocated() && customAllocated;
}

} // namespace eml

#endif // EML_TENSOR_H