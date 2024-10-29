#include "tests/Tests.h"
#include "benchmarks/benchmarks.h"

//-----------------------------------------------
// Test Bench
//-----------------------------------------------
// ................................................................................
// The test bench runs both tests and benchmarks
// ................................................................................


int main()
{
    RunTests();
    RunBenchmarks();
    return 0;
}