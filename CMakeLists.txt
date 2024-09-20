cmake_minimum_required(VERSION 3.18)

project(CudaFractal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets Concurrent OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)

qt_standard_project_setup()

set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")

file(GLOB SOURCES
    "${SRC_DIR}/*.cpp"
    "${SRC_DIR}/*.cu"
)

qt_add_executable(CudaFractal ${SOURCES})

add_custom_target(WrapGlslAsLiteral
    COMMAND ${CMAKE_COMMAND} -D SRC_DIR=${SRC_DIR} -P ${CMAKE_SOURCE_DIR}/wrap_glsl_as_literal.cmake
    COMMENT "Wrapping .glsl files..."
)
add_dependencies(CudaFractal WrapGlslAsLiteral)

set_target_properties(CudaFractal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_link_libraries(CudaFractal PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::Concurrent
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)

target_link_options(CudaFractal PRIVATE /NODEFAULTLIB:LIBCMT)