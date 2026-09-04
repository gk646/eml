#ifndef EML_TENSOR_H
#define EML_TENSOR_H

#include "eml/util/Types.h"
#include <cstring>

// ================================================================
// Tensor
// ================================================================
// ................................................................
// Class implements a 4 dimensional tensor - for data operations see math/TensorOps.h
// If not specified batch and channel are 1 - as in the tensor has 1 batch and 1 channel per default
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

    // Constructs the tensor from the given shape
    explicit Tensor( const Tuple& shape );

    // Access via flattened array index
    T& operator[]( int32_t idx );
    const T& operator[]( int32_t idx ) const;

    // ============ Memory ============

    // Allocates memory according to the current size ONLY when data is nullptr - uses PlatformAlloc()
    void allocate();

    // Frees the memory and sets data to nullptr
    void free();

    // Initializes the tensor with user managed memory
    // IMPORTANT: you need to zero initialized empty stack memory (e.g. int buff[25]{} // With {} init)
    void allocateCustom( void* memory, int32_t count );

    // Sets the data to nullptr and returns the memory
    void* freeCustom();

    // Returns true if the tensor is allocated
    [[nodiscard]] bool isAllocated() const;

    // Returns true if the tensor is allocated custom
    [[nodiscard]] bool isAllocatedCustom() const;

    // Returns a copy of the tensor using allocate() - equal to Tensor(shape()), allocate() and memcpy()
    Tensor copyTensor() const;

    T* data()
    {
        return ptr;
    }

    const T* data() const
    {
        return ptr;
    }

    // ============ Shape ============

    // Returns a tuple that contains this vectors dimensions in the form {n, c, h, w}
    [[nodiscard]] Tuple shape() const;

    // Transposes the tensor inplace
    void transpose();

    // Reshapes the tensor to the given shape
    // If already allocated, only works if the new shape fits into existing memory
    void reshape( const Tuple& shape );

    // ============ Misc ============

    // Prints the tensor with PlatformPrint()
    void print( const char* name = "Tensor" ) const;

    // ============ Data ============

    T* ptr = nullptr;     // Data pointer
    int32_t capacity = 0; // Allocated size
    int32_t size = 0;     // Maximum elements

    int32_t n = 0;        // Batch
    int32_t c = 0;        // Channels
    int32_t h = 0;        // Height / Rows
    int32_t w = 0;        // Width / Columns

    int32_t hw = 0;       // Channel stride
    int32_t chw = 0;      // Batch stride

    // ============ Metadata ============

    int32_t scale = 1;            // Scale for quantized values
    bool requiresGrad = false;    // If gradients should be computed for this tensor
    bool customAllocated = false; // If this tensor is NOT allocated with PlatformAllocate()
    Tensor<T>* grad = nullptr;    // Pointer to the grad tensor

#ifdef EML_DEBUG
    ~Tensor();
#endif
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
Tensor<T>::Tensor( const Tuple& shape ) : Tensor( shape.first, shape.second, shape.third, shape.fourth )
{
}

template <typename T>
T& Tensor<T>::operator[]( int32_t idx )
{
    EML_ASSERT( ptr != nullptr, "Tensor is not allocated!" );
    EML_ASSERT( idx < size && idx < capacity && idx >= 0, "Out of bounds access" );
#ifdef EML_DEBUG
    const int32_t batch = idx / chw;
    const int32_t channel = ( idx % chw ) / hw;
    const int32_t row = ( idx % hw ) / w;
    const int32_t col = ( idx % w );
    EML_ASSERT( batch < n && channel < c && row < h && col < w, "Out of bounds access" );
#endif
    return ptr[ idx ];
}

template <typename T>
const T& Tensor<T>::operator[]( int32_t idx ) const
{
    EML_ASSERT( idx < size && idx < capacity && idx >= 0, "Out of bounds access" );
#ifdef EML_DEBUG
    const int32_t batch = idx / chw;
    const int32_t channel = ( idx % chw ) / hw;
    const int32_t row = ( idx % hw ) / w;
    const int32_t col = ( idx % w );
    EML_ASSERT( batch < n && channel < c && row < h && col < w, "Out of bounds access" );
#endif
    return ptr[ idx ];
}

template <typename T>
void Tensor<T>::allocate()
{
    EML_ASSERT( capacity == 0, "Calling allocate() on an already allocated tensor" );
    EML_ASSERT( size > 0, "Calling allocate() on an empty tensor - likely a mistake" );
    ptr = static_cast<T*>( PlatformAlloc( sizeof( T ) * size ) );
    memset( ptr, 0, size * sizeof( T ) );
    capacity = size;
}

template <typename T>
void Tensor<T>::free()
{
    EML_ASSERT( !customAllocated, "Calling free() on custom allocated tensor! Call freeCustom()" );
    EML_ASSERT( capacity > 0, "Calling free() on empty an tensor - likely a mistake" );
    PlatformFree( ptr );
    ptr = nullptr;
    capacity = 0;
}

template <typename T>
void Tensor<T>::allocateCustom( void* memory, const int count )
{
    EML_ASSERT( static_cast<T*>( memory ) != nullptr && count > 0, "Calling stackAllocate() with invalid parameters" );
    EML_ASSERT( ptr == nullptr, "Cannot initialize an allocated tensor! Call free() or freeCustom() first" );
    EML_ASSERT( count >= size, "Passed memory is too small to hold the tensor data!" );
    ptr = static_cast<T*>( memory );
    capacity = count;
    customAllocated = true;
}

template <typename T>
void* Tensor<T>::freeCustom()
{
    EML_ASSERT( customAllocated, "Calling freeCustom() on non-custom allocated tensor! Call free()" );
    EML_ASSERT( capacity > 0, "Calling free() on an empty tensor - likely a mistake" );
    auto* temp = ptr;
    ptr = nullptr;
    capacity = 0;
    return temp;
}

template <typename T>
bool Tensor<T>::isAllocated() const
{
    return ptr != nullptr;
}

template <typename T>
bool Tensor<T>::isAllocatedCustom() const
{
    return isAllocated() && customAllocated;
}

template <typename T>
Tensor<T> Tensor<T>::copyTensor() const
{
    Tensor<T> copy{ n, c, h, w };
    copy.scale = scale;
    copy.allocate();
    memcpy( copy.ptr, ptr, size * sizeof( T ) );
    return copy;
}

template <typename T>
Tuple Tensor<T>::shape() const
{
    return { n, c, h, w };
}

template <typename T>
void Tensor<T>::transpose()
{
    if( h == w )
    {
        for( int32_t i = 0; i < h; i++ )
        {
            for( int32_t j = i + 1; j < w; j++ )
            {
                int32_t indexT = j * w + i;
                int32_t index = i * w + j;
                const auto tmp = operator[]( indexT );
                operator[]( indexT ) = operator[]( index );
                operator[]( index ) = tmp;
            }
        }
    }
    else
    {
        EML_ASSERT( false, "Not implemented" );
    }

    const int32_t tmp = h;
    h = w;
    w = tmp;
}

template <typename T>
void Tensor<T>::reshape( const Tuple& shape )
{
    const int32_t magnitudeNew = shape.first * shape.second * shape.third * shape.fourth;
    const int32_t magnitude = n * c * h * w;
    EML_ASSERT( magnitudeNew > magnitude, "New shape exceeds current bounds. Allocate a bigger tensor" );
    n = shape.first;
    c = shape.second;
    h = shape.third;
    w = shape.fourth;
}

template <typename T>
void Tensor<T>::print( const char* name ) const
{
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
                PlatformPrint( "[" );
                for( int32_t col = 0; col < w; ++col )
                {
                    const auto val = this->operator[]( rowIndex + col );
                    if constexpr( std::is_floating_point_v<T> )
                    {
                        if( static_cast<float>( static_cast<int>( val ) ) == val )
                        {
                            PlatformPrint( "%4.0f. ", val );
                        }
                        else
                        {
                            PlatformPrint( "%4.1f", val );
                        }
                    }
                    else
                    {
                        PlatformPrint( " %4d", val );
                    }
                }
                PlatformPrint( "]" );
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

#ifdef EML_DEBUG
template <typename T>
Tensor<T>::~Tensor()
{
#ifndef EML_TEST // Annoying for tests
    EML_ASSERT( data == nullptr, "Tensor wasnt freed!" );
#endif
}
#endif

} // namespace eml

#endif // EML_TENSOR_H