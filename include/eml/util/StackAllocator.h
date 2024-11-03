#ifndef EML_UTIL_STACKALLOCATOR_H
#define EML_UTIL_STACKALLOCATOR_H

namespace eml
{

struct StackAllocator final
{
    static constexpr int HISTORY_SIZE = 15;

    unsigned char* memory = nullptr; // managed memory
    int32_t capacity = 0; // Valid size of memory
    int32_t size = 0; // Current end of the stackpointer
    int32_t historySize[ HISTORY_SIZE ]{};
    int32_t historyIdx = 0;

#ifdef EML_DEBUG
    unsigned char* historyPtr[ HISTORY_SIZE ]{};
#endif

    StackAllocator( void* memory, const int32_t capacity )
        : memory( static_cast<unsigned char*>( memory ) ), capacity( capacity )
    {
        EML_ASSERT( memory != nullptr, "Passed invalid memory to initialize" );
        EML_ASSERT( capacity > 0, "Passed invalid capacity to initialize" );
    }

    void* allocate( const int32_t allocSize )
    {
        EML_ASSERT( size + allocSize < capacity, "Requested size overflows allocator" );
        EML_ASSERT(historyIdx < HISTORY_SIZE, "Too many allocation requests! Increase HISTORY_SIZE");

        historySize[ historyIdx ] = allocSize;

#ifdef EML_DEBUG
        historyPtr[ historyIdx ] = memory + size;
#endif

        ++historyIdx;
        void* ret = memory + size;
        size += allocSize;
        return ret;
    }

    void free( const void* free )
    {
        EML_ASSERT( historyIdx > 0, "Free called with no allocations made" );
        EML_ASSERT( free != nullptr, "Free called with nullptr" );

        const int allocSize = historySize[ historyIdx - 1 ];

#ifdef EML_DEBUG
        const int calcSize = (int)( memory - (unsigned char*)free ) + allocSize;
        EML_ASSERT( calcSize == size, "Freed memory in wrong order!" );
        EML_ASSERT( historyPtr[ historyIdx - 1 ] == (unsigned char*)free, "Freed memory in wrong order!" );
#endif

        size -= allocSize;
        --historyIdx;
        EML_ASSERT( size >= 0, "Freed invalid memory" );
    }
};

} // namespace eml

#endif // EML_UTIL_STACKALLOCATOR_H