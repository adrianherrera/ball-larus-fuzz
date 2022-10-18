#
# Pass
#

add_library(PathProfiling SHARED
  Transforms/Instrumentation/PathProfiling.cpp
  Analysis/PathNumbering.cpp
)
set_target_properties(PathProfiling
  PROPERTIES LINK_FLAGS "-Wl,-znodelete"
)
install(TARGETS PathProfiling LIBRARY DESTINATION lib)

#
# Runtime
#

add_library(PathProfilingRT SHARED
  Runtime/afl-compiler-rt.o.c
)
target_include_directories(PathProfilingRT
  PRIVATE "${CMAKE_SOURCE_DIR}/ext/aflplusplus/instrumentation"
)
install(TARGETS PathProfilingRT LIBRARY DESTINATION lib)
