#ifndef EML_PLATFORM_H
#define EML_PLATFORM_H

#include <cstdint>

//-----------------------------------------------
// Platform Interface
//-----------------------------------------------
// ................................................................................
// This header defines the public interface of a platform for ecml
// ................................................................................

namespace eml
{
    //----------------- SETUP -----------------//

    // Called at program startup
    void PlatformInit();

    // Called at program stop
    void PlatformDestroy();

    //----------------- IO -----------------//

    // Used for asserts
    void PlatformAssert(char const* expr, const char* msg, char const* file, unsigned line);

    // Used as stdout
    void PlatformPrint(const char* format, ...);

    // Logs a string
    void PlatformLog(const char* format, ...);

    //----------------- MEASURE -----------------//

    // Returns the current cpu clock count
    uint32_t PlatformClockCount();

    //----------------- MEMORY -----------------//

    // Handles memory allocations
    void* PlatformAlloc(uint32_t size);
    void PlatformFree(void* ptr);

    //----------------- MULTICORE -----------------//

    // Executes the given method in the specified core
    void PlatformCoreExecute(int core, void* method);

    // Sends data from core 'from' to core 'to'
    void PlatformCoreSendData(int from, int to, void* data);

    //----------------- MISC -----------------//

    // Returns a human-readable platform name
    const char* PlatformName();

} // namespace eml
#endif // EML_PLATFORM_H