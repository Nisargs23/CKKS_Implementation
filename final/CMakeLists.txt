cmake_minimum_required(VERSION 3.15)
project(CKKS_Bootstrapping_Image)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Keep single-config generators (Ninja/Makefiles) from silently using no build type.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    message(STATUS "CMAKE_BUILD_TYPE not set. Defaulting to Release")
endif()

option(CKKS_ENABLE_NATIVE_OPTIMIZATIONS "Enable CPU-native tuning flags" OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^| )-O[0-3s]( |$)")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3")
    endif()
    if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^| )-DNDEBUG( |$)")
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -DNDEBUG")
    endif()
    if(CKKS_ENABLE_NATIVE_OPTIMIZATIONS)
        if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^| )-march=native( |$)")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -march=native")
        endif()
        if(NOT CMAKE_CXX_FLAGS_RELEASE MATCHES "(^| )-mtune=native( |$)")
            string(APPEND CMAKE_CXX_FLAGS_RELEASE " -mtune=native")
        endif()
    endif()
endif()

# Find OpenMP for parallel RGB channel processing
find_package(OpenMP)

# Set OpenFHE paths manually if find_package doesn't work
set(OPENFHE_INSTALL_DIR "D:/Manipal/Research/openfhe-install" CACHE PATH "OpenFHE install directory")
set(OPENFHE_INCLUDE "${OPENFHE_INSTALL_DIR}/include/openfhe")
set(OPENFHE_LIBDIR "${OPENFHE_INSTALL_DIR}/lib")

# Try to find OpenFHE package
find_package(OpenFHE QUIET)

# If not found, set paths manually
if(NOT OpenFHE_FOUND)
    message(STATUS "OpenFHE not found via find_package, using manual paths")
    message(STATUS "OpenFHE include: ${OPENFHE_INCLUDE}")
    message(STATUS "OpenFHE lib: ${OPENFHE_LIBDIR}")
endif()

# ========================================
# Main Demo - Dummy Image with Bootstrapping
# ========================================
add_executable(ckks_bootstrap_image main.cpp image_processor.cpp)

# Include OpenFHE directories
target_include_directories(ckks_bootstrap_image PRIVATE 
    ${OPENFHE_INSTALL_DIR}/include
    ${OPENFHE_INSTALL_DIR}/include/openfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/core
    ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
    ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/third-party/include
)

# Link OpenFHE libraries
target_link_directories(ckks_bootstrap_image PRIVATE ${OPENFHE_LIBDIR})
target_link_libraries(ckks_bootstrap_image OPENFHEpke OPENFHEbinfhe OPENFHEcore)

# Set output directory
set_target_properties(ckks_bootstrap_image PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========================================
# MNIST Version - Real-world dataset
# ========================================
add_executable(ckks_bootstrap_mnist main_mnist.cpp image_processor.cpp)

# Include OpenFHE directories for MNIST version
target_include_directories(ckks_bootstrap_mnist PRIVATE 
    ${OPENFHE_INSTALL_DIR}/include
    ${OPENFHE_INSTALL_DIR}/include/openfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/core
    ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
    ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/third-party/include
)

# Link OpenFHE libraries for MNIST version
target_link_directories(ckks_bootstrap_mnist PRIVATE ${OPENFHE_LIBDIR})
target_link_libraries(ckks_bootstrap_mnist OPENFHEpke OPENFHEbinfhe OPENFHEcore)

# Set output directory for MNIST version
set_target_properties(ckks_bootstrap_mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========================================
# High-Res Neural Net - 64x64 with NN layer simulation
# ========================================
add_executable(ckks_nn_highres main_highres.cpp security_metrics.cpp)

# Include OpenFHE directories for high-res version
target_include_directories(ckks_nn_highres PRIVATE 
    ${OPENFHE_INSTALL_DIR}/include
    ${OPENFHE_INSTALL_DIR}/include/openfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/core
    ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
    ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/third-party/include
)

# Link OpenFHE libraries for high-res version
target_link_directories(ckks_nn_highres PRIVATE ${OPENFHE_LIBDIR})
target_link_libraries(ckks_nn_highres OPENFHEpke OPENFHEbinfhe OPENFHEcore)

# Set output directory for high-res version
set_target_properties(ckks_nn_highres PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========================================
# MedMNIST - Encrypted Medical Image Classification
# ========================================
add_executable(ckks_medmnist main_medmnist.cpp security_metrics.cpp)

# Include OpenFHE directories for MedMNIST version
target_include_directories(ckks_medmnist PRIVATE 
    ${OPENFHE_INSTALL_DIR}/include
    ${OPENFHE_INSTALL_DIR}/include/openfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/core
    ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
    ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/third-party/include
)

# Link OpenFHE libraries for MedMNIST version
target_link_directories(ckks_medmnist PRIVATE ${OPENFHE_LIBDIR})
target_link_libraries(ckks_medmnist OPENFHEpke OPENFHEbinfhe OPENFHEcore)

# Set output directory for MedMNIST version
set_target_properties(ckks_medmnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ========================================
# MedMNIST RGB - Encrypted RGB Image Classification
# ========================================
add_executable(ckks_medmnist_rgb main_medmnist_rgb.cpp security_metrics.cpp)

# Include OpenFHE directories for MedMNIST RGB version
target_include_directories(ckks_medmnist_rgb PRIVATE 
    ${OPENFHE_INSTALL_DIR}/include
    ${OPENFHE_INSTALL_DIR}/include/openfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/core
    ${OPENFHE_INSTALL_DIR}/include/openfhe/pke
    ${OPENFHE_INSTALL_DIR}/include/openfhe/binfhe
    ${OPENFHE_INSTALL_DIR}/include/openfhe/third-party/include
)

# Link OpenFHE libraries for MedMNIST RGB version
target_link_directories(ckks_medmnist_rgb PRIVATE ${OPENFHE_LIBDIR})
target_link_libraries(ckks_medmnist_rgb OPENFHEpke OPENFHEbinfhe OPENFHEcore)

# Enable OpenMP for parallel per-channel encryption
if(OpenMP_CXX_FOUND)
    target_link_libraries(ckks_medmnist_rgb OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found — RGB channels will process in parallel")
else()
    message(STATUS "OpenMP not found — RGB channels will process sequentially")
endif()

# Set output directory for MedMNIST RGB version
set_target_properties(ckks_medmnist_rgb PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
