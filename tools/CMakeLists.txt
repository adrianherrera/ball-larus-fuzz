find_program(CLANG_BINARY "clang" HINTS ${LLVM_TOOLS_BINARY_DIR})
find_program(CLANGXX_BINARY "clang++" HINTS ${LLVM_TOOLS_BINARY_DIR})

configure_file(path-cc.sh.in path-cc @ONLY)
configure_file(path-c++.sh.in path-c++ @ONLY)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/path-cc" TYPE BIN)
install(PROGRAMS "${CMAKE_CURRENT_BINARY_DIR}/path-c++" TYPE BIN)
