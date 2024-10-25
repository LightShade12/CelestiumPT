cmake_minimum_required(VERSION 3.13)  # CMake version check
project(imgui)                        # Create project "imgui"
set(CMAKE_CXX_STANDARD 14)            # Enable c++14 standard

# Optionally set compiler paths if needed
# set(CMAKE_C_COMPILER "/usr/local/bin/clang")
# set(CMAKE_CXX_COMPILER "/usr/local/bin/clang++")

# List all source files for the library
set(SOURCE_FILES 
    imgui.cpp 
    imgui_draw.cpp 
    imgui_demo.cpp 
    imgui_tables.cpp 
    imgui_widgets.cpp 
    backends/imgui_impl_opengl3.cpp 
    backends/imgui_impl_glfw.cpp)

# Add library target with the source files
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})

# Include directories for this project
target_include_directories(${PROJECT_NAME} PUBLIC ./ ../glfw/include)
