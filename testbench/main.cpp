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
    PlatformPrint( "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n"
                   "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM'.XMMMMM\n"
                   "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM. XMMMMM\n"
                   "MMMMMMMMNkoookNMMMMMxOMKxooxKMMKxoodKMMMMM. XMMMMM\n"
                   "MMMMMMN; :xkx; :WMMM  'cxkd, ,':dkx, ,MMMM. XMMMMM\n"
                   "MMMMMM, kMMMMM: lMMM  kMMMMM. dMMMMM. KMMM. XMMMMM\n"
                   "MMMMMW  ''''''..cMMM  WMMMMM. XMMMMM' KMMM. XMMMMM\n"
                   "MMMMMM. KMMMMMMMMMMM  WMMMMM. XMMMMM' KMMM. XMMMMM\n"
                   "MMMMMM0..dXWWNKkWMMM  WMMMMM. XMMMMM' KMMM. XMMMMM\n"
                   "MMMMMMMWk:,..';lWMMM;,WMMMMMc,NMMMMMl,XMMM:,NMMMMM\n"
                   "MMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMMM\n"
                   " ===============================================\n"
                   "|          embeddable machine learning          |\n"
                   "|                Version: %-17s    |\n"
                   " ===============================================\n\n",
                   EML_VERSION );

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
    printf( "\nTestbench took: %.2f seconds\n", ( (float)( end - start ) / 1'000 ) / 1'000.0F );
    return 0;
}