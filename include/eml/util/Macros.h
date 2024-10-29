#ifndef EML_MACROS_H
#define EML_MACROS_H


#define EML_IF_CXPR(T, type) if constexpr (std::is_same_v<T, type>)


#endif //EML_MACROS_H