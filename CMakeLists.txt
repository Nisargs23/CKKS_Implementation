cmake_minimum_required(VERSION 3.15)
project(CKKS_Bootstrapping_Image)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

# Add executable
add_executable(ckks_bootstrap_image src/main.cpp src/image_processor.cpp)

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
add_executable(ckks_bootstrap_mnist src/main_mnist.cpp src/image_processor.cpp)

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
