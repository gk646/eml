#include <eml/platform.h>

#include <cstdlib>

namespace eml
{

void PlatformInit()
{
}

void PlatformDestroy()
{
}

void PlatformAssert( char const* expr, const char* msg, char const* file, unsigned line )
{

}

void PlatformPrint( const char* format, ... )
{

}

void PlatformLog( const char* format, ... )
{

}

uint32_t PlatformCycleCount()
{
    return 0;
}

uint32_t PlatformClock()
{
    return 0;
}

void* PlatformAlloc( uint32_t size )
{
    return malloc( size );
}

void PlatformFree( void* ptr )
{
    free( ptr );
}

const char* PlatformName()
{
    return "Windows";
}

} // namespace eml