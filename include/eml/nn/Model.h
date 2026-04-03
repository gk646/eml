#ifndef EML_NN_MODEL_H
#define EML_NN_MODEL_H

#include <eml/config.h>
#include <eml/nn/LayerUtil.h>
#include <eml/nn/Optim.h>
#include <eml/util/Macros.h>

// ================================================================
// Model
// ================================================================
// ................................................................
// A model is more than just a simple container (like in pytorch)
// A model is the autograd and allocation context for all tensors within the model
// Thus a model is mandatory for autograd.
// This makes memory management straightforward and allows memory stability (no allocations after setup)
// ................................................................

namespace eml::nn
{

template <typename T>
struct Model
{
    // Creates a new model
    //    - inputShape:    shape of the input - batch size is given in the N-dimensions (N,C,H,W)!
    //    - memory:        memory the model and all its operations are allowed to use
    //    - size:          valid size of the passed memory
    //    - withTraining:  reserves memory for training data - if false training can NOT be enabled
    Model( int32_t batchSize, void* memory, int size, bool withTraining = false );

    // ============ Forward ============

    //
    virtual Tensor<T> forward( Tensor<T>& input ) = 0;

    // ============ Learning ============

    void setAutograd( bool value );

    // Calculates the gradients for all tensors that require grad
    void backward( Tensor<T>& out )
    {
    }

    void step(Optimizer optim)
    {
    }

    void zeroGrad()
    {
    }

    // ============ Info ============

    // Returns the total amount of bytes needed for this model - only available after the model is constructed
    [[nodiscard]] int32_t getMemorySize() const;

    [[nodiscard]] int32_t getWeightCount() const;

    // Returns the amount of multiplications needed for a single forward pass
    [[nodiscard]] int32_t getMultsCount() const;

  private:
    ModelContext context;      // Internal data
    int32_t batches = 1;       // How many batches to support maximum
    int32_t modelWts = 0;      // Total learnable parameters
    int32_t layerCnt = 0;      // How many layers the model has
    bool withTraining = false; // If the model is supports training
    bool isAutograd = false;   // If training is enabled

    EML_CHECK_TYPE_SUPPORTED();
    EML_FRIEND_LAYERS();
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

/*
 *  What the model needs to save:
 *      - Tensor Data
 *          - Weights, Biases
 *          - Created during forward pass
 *      - Autograd
 *          - Computation Graph Nodes
 *          - Grad data for each weight and bias tensor for each batch
 *      - Indexing Data
 */

template <typename T>
Model<T>::Model( const int32_t batchSize, void* memory, int size, const bool withTraining )
    : context( memory, size, batchSize ), batches( batchSize ), withTraining( withTraining )
{
    EML_ASSERT( batchSize > 0, "Batch size must be greater than zero" );
    printf( "Model Base\n" );
}

template <typename T>
void Model<T>::setAutograd( const bool value )
{
    EML_ASSERT( value == false || withTraining == true, "Model is not configured for training" );
    isAutograd = value;
}

template <typename T>
int32_t Model<T>::getMemorySize() const
{
    return context.layerOff;
}

template <typename T>
int32_t Model<T>::getWeightCount() const
{
}

template <typename T>
int32_t Model<T>::getMultsCount() const
{
}

} // namespace eml::nn

#endif // EML_NN_MODEL_H