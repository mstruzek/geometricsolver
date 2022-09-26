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
        LANGUAGES CXX C CUDA)



set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES 75)





if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CUDAFLAGS} -Xcompiler \"/Zc:__cplusplus\"")
endif()



set(PROJECT_SHARED_LIBRARY libgsketcherjni)
set(PROJECT_TEST_EXECUTABLE gsketcherjnie)

#
set(TEST_EXECUTABLE OFF)


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
		src/solver_kernel.cu
		src/format_encoder.cu
		src/model.cu
		src/quda.cpp
        src/cuerror.cpp		
        src/settings.cpp
		src/model_config.cpp
		src/gpu_utility.cpp
        src/gpu_computation.cpp
        src/gpu_geometric_solver.cpp
        src/gpu_computation_context.cpp	
        src/gpu_solver_system.cpp
        src/solver/gpu_solver_dense_lu.cpp        
        src/solver/gpu_solver_sparse_qr.cpp
		src/solver/gpu_solver_sparse_bicgstab.cpp
		src/solver/gpu_sparse_precondition_ilu02.cpp
        src/tensor_operation.cpp				
        src/com_mstruzek_jni_JNISolverGate.cpp
		)





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

if(TEST_EXECUTABLE)

add_executable(${PROJECT_TEST_EXECUTABLE} src/entry.cpp 	# main()
		src/solver_kernel.cu
		src/format_encoder.cu
		src/model.cu
		src/quda.cpp
        src/cuerror.cpp
		src/model_config.cpp
        src/settings.cpp
		src/gpu_utility.cpp
        src/gpu_computation.cpp
        src/gpu_geometric_solver.cpp
        src/gpu_computation_context.cpp	
        src/gpu_solver_system.cpp
        src/solver/gpu_solver_dense_lu.cpp        
        src/solver/gpu_solver_sparse_qr.cpp
		src/solver/gpu_solver_sparse_bicgstab.cpp
		src/solver/gpu_sparse_precondition_ilu02.cpp		
		src/mmio/mmio.c
		src/mmio/mmio_wrapper.cpp										
		src/solver/test_gpu_solver_biccgstab.cpp		# main()
        src/tensor_operation.cpp		
        src/com_mstruzek_jni_JNISolverGate.cpp	
		)


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

target_include_directories(${PROJECT_TEST_EXECUTABLE} PUBLIC 
        CUDA::cudart 
        CUDA::cublas 
        CUDA::cusolver
        CUDA::cusparse)

endif()