#ifndef EML_MEMORY_H
#define EML_MEMORY_H

#include "eml/platform.h"
#include <memory_resource>

namespace eml
{

struct EmlAllocator : std::pmr::memory_resource
{
  private:
    void* do_allocate( size_t bytes, size_t alignment ) override
    {
        (void)alignment;
        return PlatformAlloc( bytes );
    }

    void do_deallocate( void* p, size_t bytes, size_t alignment ) override
    {
        (void)alignment;
        (void)bytes;
        PlatformFree( p );
    }

    [[nodiscard]] bool do_is_equal( const memory_resource& other ) const noexcept override
    {
        return this == &other;
    }
};

inline EmlAllocator ALLOCATOR{};

template<typename T>
struct AllocatorWrapper : std::pmr::polymorphic_allocator<T>
{
    AllocatorWrapper() : std::pmr::polymorphic_allocator<T>( &ALLOCATOR )
    {
    }
};

template <typename T>
using vector = std::vector<T>;

using TensorList = vector<Tensor<float>>;

} // namespace eml

#endif // EML_MEMORY_H
