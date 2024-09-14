cmake_minimum_required(VERSION 3.18)

project(CudaFractal LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)

find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets OpenGL OpenGLWidgets)
find_package(OpenGL REQUIRED)

qt_standard_project_setup()

function(make_includable input_file output_file)
    file(READ ${input_file} content)
    set(content "R\"(\n${content})\"")
    file(WRITE ${output_file} "${content}")
endfunction(make_includable)

make_includable(src/shaders/fragment.glsl src/shaders/fragment.str)
make_includable(src/shaders/vertex.glsl src/shaders/vertex.str)

qt_add_executable(CudaFractal
    src/main.cpp
    src/OpenGLWidget.cpp src/OpenGLWidget.h
    src/ShaderSrc.h
    src/cuda_kernels.cu
)

set_target_properties(CudaFractal PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES native
)

target_link_libraries(CudaFractal PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    OpenGL::GL
)
