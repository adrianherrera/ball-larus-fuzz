# Ball-Larus Path Coverage

The [Ball-Larus path profiling](https://dl.acm.org/doi/10.5555/243846.243857) algorithm is a well-known, low-cost approach for measuring path coverage.
This repo adapts the algorithm to work as a coverage feedback mechanism in [AFL++](https://github.com/AFLplusplus/AFLplusplus/), a state-of-the-art greybox fuzzer.
We build on top of an old implementation of the Ball-Larus algorithm for [LLVM](https://github.com/syoyo/LLVM/blob/master/lib/Transforms/Instrumentation/PathProfiling.cpp).
However, we adjust the implementation to work across multiple compilation units using techniques from LLVM's [sanitizer coverage](https://clang.llvm.org/docs/SanitizerCoverage.html).
We also track paths in fixed-size arrays; we do not use dynamically-constructed hash tables (as proposed in the original MICRO96 paper).

On the fuzzing end, we do not use a fixed-size bitmap.
Instead, that bitmap size is adjusted so that all paths can be stored without the need of a hashing function.
Thus, paths are exact and the coverage map is non-lossy.
If the number of paths (across all compilation units) is too large, the user can specify a threshold that leaves functions with too many paths uninstrumented.
