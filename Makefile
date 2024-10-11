# Compiler and flags
CXX := g++
NVCC := nvcc

# CUDA architecture for RTX 2070 (compute capability 7.5)
CUDA_ARCH := -arch=sm_75

# Compiler flags
CXXFLAGS := -std=c++17
NVCCFLAGS := $(CUDA_ARCH) -std=c++17 -rdc=true -lcudadevrt -lineinfo

# Debug and Optimization flags
DEBUG_FLAGS := -g 
OPT_FLAGS := -O3

# Directories
SRC_DIR := src
OBJ_DIR := obj
BIN_DIR := bin

# Files
MAIN_CPP := main.cpp # This is in the main directory
CPP_SOURCES := $(wildcard $(SRC_DIR)/*.cpp)
CU_SOURCES := $(wildcard $(SRC_DIR)/*.cu)

# Object files
MAIN_OBJECT := $(OBJ_DIR)/main.o
CPP_OBJECTS := $(patsubst $(SRC_DIR)/%.cpp, $(OBJ_DIR)/%.o, $(CPP_SOURCES))
CU_OBJECTS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(CU_SOURCES))

# Output executable name
TARGET := $(BIN_DIR)/main

# Libraries to link
LIBS := -lcublas

# Default rule
all: clean debug

# Rule for building with optimization
build: CXXFLAGS += $(OPT_FLAGS)
build: NVCCFLAGS += $(OPT_FLAGS)
build: $(TARGET)

# Rule for building with debug flags
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: NVCCFLAGS += $(DEBUG_FLAGS)
debug: $(TARGET)

# Create output binary from object files
$(TARGET): $(MAIN_OBJECT) $(CPP_OBJECTS) $(CU_OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(NVCC) $(MAIN_OBJECT) $(CPP_OBJECTS) $(CU_OBJECTS) $(LIBS) $(NVCCFLAGS) -o $@

# Rule for compiling main.cpp (in the main directory)
$(OBJ_DIR)/main.o: $(MAIN_CPP)
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -Isrc/ -c $< -o $@

# Rule for compiling C++ source files in src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -Isrc/ -c $< -o $@

# Rule for compiling CUDA source files in src/
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -Isrc/ -c $< -o $@

# Clean up compiled files
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

.PHONY: all build debug clean
