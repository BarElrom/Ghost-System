# GHOST v2 — CSI Injection Loopback firmware
cmake_minimum_required(VERSION 3.5)
add_compile_options(-fdiagnostics-color=always)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(csi_inject_recv)
