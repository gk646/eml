#ifndef EML_PLATFORM_H
#define EML_PLATFORM_H

#include <cstdint>

//===============================================
// Platform Interface
//===============================================
// ................................................................................
// This header defines the public interface of a platform for ecml
// ................................................................................

namespace eml
{
//================= SETUP =================//

// Called at program startup
void PlatformInit();

// Called at program stop
void PlatformDestroy();

//================= IO =================//

// Used for asserts
void PlatformAssert( char const* expr, const char* msg, char const* file, unsigned line );

// Used as stdout
void PlatformPrint( const char* format, ... );

// Logs a string
void PlatformLog( const char* format, ... );

//================= MEASURE =================//

// Returns the current cpu clock count
uint32_t PlatformCycleCount();

// Returns a steady clock value in microseconds
uint32_t PlatformClock();

//================= MEMORY =================//

// Allocates memory that's valid until free is called
void* PlatformAlloc( uint32_t size );

// Can be called at anytime with a pointer acquired from PlatformAlloc()
void PlatformFree( void* ptr );

//================= MULTICORE =================//

// Executes the given method on the specified core
void PlatformCoreExecute( int core, void* method );

// Sends data from core 'from' to core 'to'
void PlatformCoreSendData( int from, int to, void* data );

//================= MISC =================//

// Returns a human-readable platform name
const char* PlatformName();

} // namespace eml
#endif // EML_PLATFORM_H