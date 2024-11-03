#include <eml/platform.h>

#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>

#include <x86intrin.h>

namespace eml
{

void PlatformInit()
{
    printf( "Platform [%s] initialized", PlatformName() );
}

void PlatformDestroy()
{
    printf( "Platform [%s] destroyed", PlatformName() );
}

void PlatformAssert( char const* expr, const char* msg, char const* file, unsigned line )
{
    fprintf( stderr, "Assert failed: %s | Message: %s\nAt: %s:%d\n", expr, msg, file, line );
    abort();
}

void PlatformPrint( const char* format, ... )
{
    va_list args;
    va_start( args, format );
    vprintf( format, args );
    va_end( args );
}

void PlatformLog( const char* format, ... )
{
    va_list args;
    va_start( args, format );
    vprintf( format, args );
    va_end( args );
}

uint32_t PlatformCycleCount()
{
    return __rdtsc();
}

uint32_t PlatformClock()
{
    using namespace std::chrono;
    return std::chrono::duration_cast<microseconds>( steady_clock::now().time_since_epoch() ).count();
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