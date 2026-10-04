cmake_minimum_required(VERSION 3.20)
project(cllm_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 内核静态库：自动收集 src/ 下所有源文件（core + server）
file(GLOB_RECURSE CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)
add_library(cllm_core STATIC ${CORE_SOURCES})
target_include_directories(cllm_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

# 可执行文件
add_executable(cllm_inference main.cpp)
target_link_libraries(cllm_inference PRIVATE cllm_core)

# 线程支持（Linux 下需显式链接 pthread，macOS 无害）
find_package(Threads REQUIRED)
target_link_libraries(cllm_inference PRIVATE Threads::Threads)

# ---- 安装规则 ----
include(GNUInstallDirs)

# 安装可执行文件到 <prefix>/bin
install(TARGETS cllm_inference RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# 安装头文件到 <prefix>/include（供二次开发）
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
