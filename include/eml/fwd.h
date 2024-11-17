#ifndef EML_FWD_HPP
#define EML_FWD_HPP

#include "config.h"

namespace eml
{
template <typename T>
struct Tensor;

namespace nn
{
template <typename T>
struct Model;
template <typename T>
struct Linear;
template <typename T>
struct ReLU;
template <typename T>
struct Conv2D;
template <typename T>
struct Conv1D;
} // namespace nn

} // namespace eml

#endif // EML_FWD_HPP