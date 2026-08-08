CC = g++
CFLAGS = -O2 -fopenmp -mavx2 -mfma -mavx512f -msse 

BUILD_DIR = build

SRC = src/*.cpp \
	src/gemm/*.cpp 

all: 
	@mkdir $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(BUILD_DIR)/gemm.exe
	@./$(BUILD_DIR)/gemm.exe

clean: 
	@rm -rf $(BUILD_DIR)