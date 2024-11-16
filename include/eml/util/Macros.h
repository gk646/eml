#ifndef EML_MACROS_H
#define EML_MACROS_H




#define EML_DEFINE_IS_LAYER_TRAIT(trait_name, LayerType) \
template <typename T> \
struct trait_name : std::false_type {}; \
\
template <typename T> \
struct trait_name<LayerType<T>> : std::true_type {};




#endif //EML_MACROS_H