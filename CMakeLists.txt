cmake_minimum_required (VERSION 3.5.1)
  
project(Model)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_PREFIX_PATH "~/mylibs")
find_package(SEAL 3.5 REQUIRED)
find_package(OpenMP)

#########################################  MNIST Model ######################################### 

###Cryptonets
file(GLOB minist_cryptonets_files "${PROJECT_SOURCE_DIR}/model/Mnist/Cryptonets/*.cpp" "${PROJECT_SOURCE_DIR}/model/MNIST/Cryptonets/*.h")
add_executable(MnistCryptonets ${minist_cryptonets_files})
target_link_libraries( MnistCryptonets PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistCryptonets PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistCryptonets  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###LoLa
file(GLOB minist_lola_files "${PROJECT_SOURCE_DIR}/model/Mnist/LoLa/*.cpp")
add_executable(MnistLoLa ${minist_lola_files})
target_link_libraries( MnistLoLa PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistLoLa PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistLoLa  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-1
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/1/*.cpp")
add_executable(MnistNoya ${mnist_noya_files})
target_link_libraries( MnistNoya PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-2
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/2/*.cpp")
add_executable(MnistNoya2 ${mnist_noya_files})
target_link_libraries( MnistNoya2 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya2 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya2  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-4
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/4/*.cpp")
add_executable(MnistNoya4 ${mnist_noya_files})
target_link_libraries( MnistNoya4 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya4 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya4  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-8
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/8/*.cpp")
add_executable(MnistNoya8 ${mnist_noya_files})
target_link_libraries( MnistNoya8 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya8 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya8  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-16
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/16/*.cpp")
add_executable(MnistNoya16 ${mnist_noya_files})
target_link_libraries( MnistNoya16 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya16 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya16  PUBLIC   ${PROJECT_SOURCE_DIR}/include)


###Noya-32
file(GLOB mnist_noya_files "${PROJECT_SOURCE_DIR}/model/Mnist/Noya/32/*.cpp")
add_executable(MnistNoya32 ${mnist_noya_files})
target_link_libraries( MnistNoya32 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(MnistNoya32 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(MnistNoya32  PUBLIC   ${PROJECT_SOURCE_DIR}/include)
######################################### CIFAR Model ######################################### 

###LoLa
file(GLOB cifar_lola_files "${PROJECT_SOURCE_DIR}/model/Cifar/LoLa/*.cpp")
add_executable(CifarLoLa ${cifar_lola_files})
target_link_libraries( CifarLoLa PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarLoLa PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarLoLa  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-1
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/1/*.cpp")
add_executable(CifarNoya ${cifar_noya_files})
target_link_libraries( CifarNoya PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya-2
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/2/*.cpp")
add_executable(CifarNoya2 ${cifar_noya_files})
target_link_libraries( CifarNoya2 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya2 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya2  PUBLIC   ${PROJECT_SOURCE_DIR}/include)


###Noya-4
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/4/*.cpp")
add_executable(CifarNoya4 ${cifar_noya_files})
target_link_libraries( CifarNoya4 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya4 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya4  PUBLIC   ${PROJECT_SOURCE_DIR}/include)


###Noya-8
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/8/*.cpp")
add_executable(CifarNoya8 ${cifar_noya_files})
target_link_libraries( CifarNoya8 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya8 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya8  PUBLIC   ${PROJECT_SOURCE_DIR}/include)


###Noya-16
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/16/*.cpp")
add_executable(CifarNoya16 ${cifar_noya_files})
target_link_libraries( CifarNoya16 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya16 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya16  PUBLIC   ${PROJECT_SOURCE_DIR}/include)


###Noya-32
file(GLOB cifar_noya_files "${PROJECT_SOURCE_DIR}/model/Cifar/Noya/32/*.cpp")
add_executable(CifarNoya32 ${cifar_noya_files})
target_link_libraries( CifarNoya32 PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(CifarNoya32 PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(CifarNoya32  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###Noya for test only
file(GLOB cifar_noya_files_test "${PROJECT_SOURCE_DIR}/test/Cifar/*.cpp")
add_executable(TestCifarNoya ${cifar_noya_files_test})
target_link_libraries( TestCifarNoya PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TestCifarNoya PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(TestCifarNoya  PUBLIC   ${PROJECT_SOURCE_DIR}/include)

###SRHE for test only
file(GLOB srhe_files_test "${PROJECT_SOURCE_DIR}/test/SRHE/*.cpp")
add_executable(TestSRHE ${srhe_files_test})
target_link_libraries( TestSRHE PUBLIC SEAL::seal)
if(OpenMP_CXX_FOUND)
    target_link_libraries(TestSRHE PUBLIC OpenMP::OpenMP_CXX)
endif()
target_include_directories(TestSRHE  PUBLIC   ${PROJECT_SOURCE_DIR}/include)
