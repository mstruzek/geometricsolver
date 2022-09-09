cmake_minimum_required(VERSION 3.20)


# cuda compiler tooling


if(UNIX) 
	set(CMAKE_CUDA_COMPILER /usr/local/cuda-11.7/bin/nvcc)	
elseif(WIN32)
	set(CMAKE_CUDA_COMPILER "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.7\\bin\\nvcc.exe")
endif()



# JNI binding into cuda solver infrastructure

project(gsketcherjni VERSION 0.0.1
        DESCRIPTION "jni interface into GPU Cuda API"
        LANGUAGES CXX CUDA)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CMAKE_CUDA_ARCHITECTURES 75)



set(PROJECT_SHARED_LIBRARY libgsketcherjni)
set(PROJECT_TEST_EXECUTABLE gsketcherjnie)



# Local test configurations:    ://      NVIDIA-SMI 515.48.07    ://      Driver Version: 515.48.07       ://      CUDA Version: 11.7



###########################################     \\\library  runtime image

if(UNIX) 
	set(JAVA_HOME /usr/lib/jvm/java-17-openjdk-17.0.3.0.5-0.1.ea.el8.x86_64)	
elseif(WIN32)
	#set(JAVA_HOME "C:\\Program Files\\Java\\jdk-17.0.2\\")
endif()

find_package(JNI)     #  in this cmake-3.20 settings JNI module is not setting the target JNI::JNI (cmake-3.24)

find_package(CUDAToolkit 11.7)

add_library(${PROJECT_SHARED_LIBRARY} SHARED
        src/com_mstruzek_jni_JNISolverGate.cpp
        src/cuerror.cpp
        src/settings.cpp
        src/model.cpp
        src/gpu_geometric_solver.cpp
        src/gpu_computation_context.cpp
		src/solver_kernel.cu
        src/gpu_computation.cu
        src/tensor_operation.cu
        src/gpu_linear_system.cpp )


target_include_directories(${PROJECT_SHARED_LIBRARY} PUBLIC     
        ${JAVA_INCLUDE_PATH} 
        ${JAVA_INCLUDE_PATH2})


target_link_libraries(${PROJECT_SHARED_LIBRARY} PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cusparse)
target_include_directories(${PROJECT_SHARED_LIBRARY} PRIVATE PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cusparse)


############################################     \\\entry test executable file        

add_executable(${PROJECT_TEST_EXECUTABLE} src/entry.cpp )


target_include_directories(${PROJECT_TEST_EXECUTABLE} PUBLIC 
        JNI::JNI)
target_include_directories(${PROJECT_TEST_EXECUTABLE} PUBLIC 
        ${JAVA_INCLUDE_PATH} 
        ${JAVA_INCLUDE_PATH2})

target_link_libraries(${PROJECT_TEST_EXECUTABLE} PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cusparse)
target_include_directories(${PROJECT_TEST_EXECUTABLE} PRIVATE PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cusparse)

target_link_libraries(${PROJECT_TEST_EXECUTABLE} PUBLIC ${PROJECT_SHARED_LIBRARY})
