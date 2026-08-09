CC = g++
CFLAGS = -O3 -fopenmp -msse -mavx2 -mfma

BUILD_DIR = build

SRC = src/*.cpp \
	src/gemm/*.cpp 

all: 
	@mkdir $(BUILD_DIR)
	$(CC) $(CFLAGS) $(SRC) -o $(BUILD_DIR)/gemm.exe
	@./$(BUILD_DIR)/gemm.exe

clean: 
	@rm -rf $(BUILD_DIR)