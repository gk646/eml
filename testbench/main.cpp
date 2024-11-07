
#include "benchmarks/Benchmarks.h"
#include "tests/Tests.h"

//-----------------------------------------------
// Test Bench
//-----------------------------------------------
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
                   "|                Version: %-18s    |\n"
                   " ===============================================\n\n",
                   EML_VERSION, __DATE__, __TIME__ );

    RunTests();
    // -----------
    RunBenchmarks();
    return 0;
}