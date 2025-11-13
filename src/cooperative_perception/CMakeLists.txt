cmake_minimum_required(VERSION 3.10)
project(cooperative_perception)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
    cooperative_perception/encoder_node.py
    cooperative_perception/fusion_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages
)

ament_package()

