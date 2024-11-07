#ifndef EML_LAYERUTIL_H
#define EML_LAYERUTIL_H

#include <eml/nn/layers/Conv2D.h>

// ----------------------------------------------------------------
// LayerUtil
// ----------------------------------------------------------------
// ................................................................
// Allows to retrieve information about layers such as amount of weights, operations needed
// ................................................................

namespace eml::nn
{

// Returns the number of multiplications needed in this layer
template <typename T>
int32_t GetLayerOps( const T& layer );

}




namespace eml::nn
{

template <typename Layer, typename T>
int32_t GetLayerOps( const Layer& layer )
{
  if constexpr ( std::is_same_v<T, Conv2D<T>> )
  {

  }
}

}

#endif // EML_LAYERUTIL_H