#ifndef EML_NN_MODEL_H
#define EML_NN_MODEL_H

#include "eml/nn/AutoGrad.h"
#include "eml/nn/Optim.h"
#include <eml/config.h>

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
    Model( int32_t batchSize, bool withTraining );

    virtual ~Model() = default;

    // ============ Forward ============

    virtual TensorList forward( const TensorList& inputs );

    // ============ Learning ============

    void setAutograd( bool value );

    // Calculates the gradients for all tensors that require grad
    void backward( const TensorList& out );

    void step( Optimizer optim );

    void zeroGrad();

    void addLayer( Layer* layer )
    {
        layers.push_back( layer );
    }

    // ============ Info ============

    // Gets the total amount of weights
    uint64_t getWeightCount() const;

    // Gets the required memory for the model
    uint64_t getMemorySize() const;

    vector<Layer*> layers;
    ModelContext context;      // Internal data
    bool withAutograd = false; // If training is enabled
};

} // namespace eml::nn

// IMPLEMENTATION

namespace eml::nn
{

template <typename T>
inline Model<T>::Model( const int32_t batchSize, const bool withTraining ) : withAutograd( withTraining )
{
    EML_ASSERT( batchSize > 0, "Batch size must be greater than zero" );
}

template <typename T>
TensorList Model<T>::forward( const TensorList& in )
{
    TensorList input = in;
    TensorList outputs;
    for( auto& layer : layers )
    {
        layer->forwardI( input, outputs );
        input = outputs;
    }
    return outputs;
}
template <typename T>
void Model<T>::setAutograd( const bool value )
{
    withAutograd = value;
}

template <typename T>
void Model<T>::backward( const TensorList& out )
{
}

template <typename T>
void Model<T>::step( Optimizer optim )
{
}

template <typename T>
inline void Model<T>::zeroGrad()
{
}

template <typename T>
uint64_t Model<T>::getWeightCount() const
{
    uint64_t count = 0;
    for( auto& layer : layers )
        count += layer->getWeightCount();
    return count;
}

template <typename T>
uint64_t Model<T>::getMemorySize() const
{

    uint64_t count = 0;
    for( auto& layer : layers )
        count += layer->getMemoryUsage();
    return count;
}

} // namespace eml::nn

#endif // EML_NN_MODEL_H