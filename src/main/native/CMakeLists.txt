cmake_minimum_required(VERSION 3.20.0)

set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)

#set(JAVA_AWT_LIBRARY /usr/lib/jvm/java-17-openjdk-17.0.3.0.5-0.1.ea.el8.x86_64/lib/libawt.so)
#set(JAVA_JVM_LIBRARY /usr/lib/jvm/java-17-openjdk-17.0.3.0.5-0.1.ea.el8.x86_64/lib/libjava.so)


# Load default CUDA library into project definition

project(msketchjni VERSION 0.0.1
        DESCRIPTION "jni interface into GPU Cuda API"
        LANGUAGES CXX CUDA)

# Local test configurations:
#   NVIDIA-SMI 515.48.07
#   Driver Version: 515.48.07
#   CUDA Version: 11.7


add_library(msketchjni SHARED src/libmain.cpp)



target_include_directories(msketchjni PUBLIC AFTER
        /usr/lib/jvm/java-17-openjdk-17.0.3.0.5-0.1.ea.el8.x86_64/include/
        /usr/lib/jvm/java-17-openjdk-17.0.3.0.5-0.1.ea.el8.x86_64/include/linux/)

