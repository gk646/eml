#ifndef EML_NN_MODEL_H
#define EML_NN_MODEL_H

#include <eml/nn/LayerUtil.h>
#include <eml/util/Macros.h>

// ================================================================
// Model
// ================================================================
// ................................................................
// A model etorch
// A model is the mandatory context for autograd
// Layers can be used as standalone modules only with no autograd
//
// ................................................................
namespace eml::nn
{
struct Model final
{

  private:
    Tuple shape{};
    int32_t modelMults = 0;   // Total multiplications needed for a single inference pass
    int32_t modelWeights = 0; // Total learnable parameters

    template <class Layer>
    void registerLayer( Layer layer );


    void addAutoGradNode();
};

} // namespace eml::nn

// namespace eml::nn
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

template <class Layer>
void Model::registerLayer( Layer layer )
{
    modelMults += GetLayerOps( layer );
    modelWeights += GetLayerWeights( layer );
}

} // namespace eml::nn

#endif // EML_NN_MODEL_H