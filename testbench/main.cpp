#include "benchmarks/Benchmarks.h"
#include "tests/Tests.h"

//===============================================
// Test Bench
//===============================================
// ................................................................................
// The test bench runs both tests and benchmarks
// ................................................................................

int main()
{

    PlatformPrint( R"(MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'.XMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM. XMMMMM
MMMMMMMMNkoookNMMMMMxOMKxooxKMMKxoodKMMMMM. XMMMMM
MMMMMMN; :xkx; :WMMM  'cxkd, ,':dkx, ,MMMM. XMMMMM
MMMMMM, kMMMMM: lMMM  kMMMMM. dMMMMM. KMMM. XMMMMM
MMMMMW  ''''''..cMMM  WMMMMM. XMMMMM' KMMM. XMMMMM
MMMMMM. KMMMMMMMMMMM  WMMMMM. XMMMMM' KMMM. XMMMMM
MMMMMM0..dXWWNKkWMMM  WMMMMM. XMMMMM' KMMM. XMMMMM
MMMMMMMWk:,..';lWMMM;,WMMMMMc,NMMMMMl,XMMM:,NMMMMM
MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM
 ===============================================
|          embeddable machine learning          |
|                Version: %-17s     |
 ===============================================
)",
                   EML_VERSION_STRING );

    // =========== //
    const uint32_t start = PlatformClock();
    // =========== //
    RunTests();
    // =========== //
#ifdef NDEBUG
    RunBenchmarks();
#endif
    // =========== //
    const uint32_t end = PlatformClock();
    // =========== //
    printf( "\nTestbench took: %.2f seconds", ( (float)( end - start ) / 1'000 ) / 1'000.0F );
    return 0;
}