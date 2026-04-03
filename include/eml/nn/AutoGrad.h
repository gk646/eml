#ifndef EML_NN_AUTOGRAD_H
#define EML_NN_AUTOGRAD_H

#include <eml/fwd.h>
#include <eml/nn/Layer.h>
#include <eml/util/Types.h>

// ================================================================
// AutoGrad
// ================================================================
// ................................................................
// ................................................................

namespace eml
{

using GradIndex = uint8_t;

enum class Operation : uint8_t
{
    // Tensor
    TEN_MATMUL,
    TEN_MATMUL_A_TRANS,
    TEN_MATMUL_B_TRANS,
    TEN_E_ADD,

    // Activation Function
    ACTF_RELU,

    // Loss function
    LOSS_MSE,
};

template <typename T>
struct AutoGradNode final
{
    static constexpr int8_t MAX_INPUTS = 3;
    static constexpr int8_t MAX_OUTPUTS = 2;
    Operation op; // operation this tensor was created with
    Layer& layer;
    Tensor<T>* inputs[ MAX_INPUTS ];
    Tensor<T>* outputs[ MAX_INPUTS ];
};

// Initialization order for classes is:
//      - members first then constructor
//      - base class before derived class

/* Data layout
 *|----------------------------------------------------------------------------------------------------------|
 *|/////////////////////////////////////|////////////////////|//////////////////////////|////////////////////|
 *|//////////// Layer Data /////////////|// Local Tensors ///|///// Autograd Nodes /////|// Layer Pointers //|
 *|/////////////////////////////////////|////////////////////|//////////////////////////|////////////////////|
 *|----------------------------------------------------------------------------------------------------------|
 * -> ...                               |reserved at register|layerCount * sizeof(Node) |               ... <-
 */

struct ModelContext
{
    unsigned char* mem = nullptr;
    int32_t memCnt = 0;
    int32_t batches = 0;

    // ============ Info ============

    int32_t layerOff = 0;
    int32_t pointerOff = 0; // Beginning of the: 'Layer Pointers' partition
    int32_t tempTensors = 0; // Size of the: 'Local Tensor' partition

    ModelContext( void* memory, const int32_t size, const int32_t batchSize )
        : mem( (unsigned char*)memory ), memCnt( size ), batches( batchSize )
    {

    }

    void* requestLayerMemory( const int32_t cnt, const int32_t size )
    {
        const int32_t requested = cnt * size;
#ifdef EML_DEBUG
        if( layerOff + requested >= memCnt )
        {
            PlatformPrint( "Model memory not sufficient! At least needed: %d", layerOff + requested + 1 );
            EML_ASSERT( false, "Model memory not sufficient!" );
        }
#endif
        const auto ret = ( mem + layerOff );
        layerOff += requested;
        return ret;
    }

    void registerLayer( Layer& layer )
    {
        if(layerOff >( memCnt -  pointerOff))
        {

        }
    }


};

} // namespace eml

#endif // EML_NN_AUTOGRAD_H