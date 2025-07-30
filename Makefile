# Compiler and assembler settings
NASM = nasm
CC = gcc
AR = ar

# Architecture-specific flags
ASM_FLAGS = -f elf64 -g -F dwarf
CC_FLAGS = -std=c99 -Wall -Wextra -O3 -mavx2 -mfma -g
LD_FLAGS = -lm

# Directories
SRC_DIR = src
TEST_DIR = tests
DOCS_DIR = docs
BUILD_DIR = build
LIB_DIR = lib
INCLUDE_DIR = include

# Source files
ASM_SOURCES = $(wildcard $(SRC_DIR)/**/*.asm) $(wildcard $(SRC_DIR)/*.asm)
C_SOURCES = $(wildcard $(SRC_DIR)/**/*.c) $(wildcard $(SRC_DIR)/*.c)
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)

# Object files
ASM_OBJECTS = $(ASM_SOURCES:$(SRC_DIR)/%.asm=$(BUILD_DIR)/%.o)
C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/%.o)
SHARED_ASM_OBJECTS = $(ASM_SOURCES:$(SRC_DIR)/%.asm=$(BUILD_DIR)/shared/%.o)
SHARED_C_OBJECTS = $(C_SOURCES:$(SRC_DIR)/%.c=$(BUILD_DIR)/shared/%.o)
TEST_OBJECTS = $(TEST_SOURCES:$(TEST_DIR)/%.c=$(BUILD_DIR)/tests/%.o)

# Library and executable names
STATIC_LIB = $(LIB_DIR)/libmlasm.a
SHARED_LIB = $(LIB_DIR)/libmlasm.so
TEST_RUNNER = $(BUILD_DIR)/test_runner
UNIT_TESTS = $(BUILD_DIR)/unit_tests
BENCHMARK = $(BUILD_DIR)/benchmark

# Detect CPU features
CPU_FEATURES := $(shell lscpu | grep -i flags)
ifneq (,$(findstring avx512,$(CPU_FEATURES)))
    ASM_FLAGS += -DAVX512_SUPPORT
    CC_FLAGS += -mavx512f -mavx512dq
endif

# Default target
all: directories $(STATIC_LIB) $(TEST_RUNNER) $(UNIT_TESTS) $(BENCHMARK)

# Build with shared library
all-shared: directories $(STATIC_LIB) $(SHARED_LIB) $(TEST_RUNNER) $(UNIT_TESTS) $(BENCHMARK)

# Create necessary directories
directories:
	@mkdir -p $(BUILD_DIR)/vectors $(BUILD_DIR)/models $(BUILD_DIR)/activations $(BUILD_DIR)/utils $(BUILD_DIR)/tests
	@mkdir -p $(BUILD_DIR)/shared/vectors $(BUILD_DIR)/shared/models $(BUILD_DIR)/shared/activations $(BUILD_DIR)/shared/utils
	@mkdir -p $(LIB_DIR) $(INCLUDE_DIR)

# Compile assembly sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.asm
	$(NASM) $(ASM_FLAGS) $< -o $@

# Compile assembly sources for shared library
$(BUILD_DIR)/shared/%.o: $(SRC_DIR)/%.asm
	$(NASM) $(ASM_FLAGS) $< -o $@

# Compile C sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Compile C sources for shared library
$(BUILD_DIR)/shared/%.o: $(SRC_DIR)/%.c
	$(CC) $(CC_FLAGS) -fPIC -I$(INCLUDE_DIR) -c $< -o $@

# Compile test sources
$(BUILD_DIR)/tests/%.o: $(TEST_DIR)/%.c
	$(CC) $(CC_FLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Create static library
$(STATIC_LIB): $(ASM_OBJECTS) $(C_OBJECTS)
	$(AR) rcs $@ $^

# Create shared library
$(SHARED_LIB): $(SHARED_ASM_OBJECTS) $(SHARED_C_OBJECTS)
	$(CC) -shared -o $@ $^ $(LD_FLAGS)

# Build test runner (only test_main.c)
$(TEST_RUNNER): $(BUILD_DIR)/tests/test_main.o $(STATIC_LIB)
	$(CC) -no-pie -o $@ $^ $(LD_FLAGS)

# Build unit tests (only unit_tests.c)
$(UNIT_TESTS): $(BUILD_DIR)/tests/unit_tests.o $(STATIC_LIB)
	$(CC) -no-pie -o $@ $^ $(LD_FLAGS)

# Build benchmark suite
$(BENCHMARK): $(BUILD_DIR)/tests/benchmark.o $(STATIC_LIB)
	$(CC) -no-pie -o $@ $^ $(LD_FLAGS)

# Install headers
install-headers:
	cp -r $(INCLUDE_DIR)/* /usr/local/include/

# Install libraries
install-libs: $(STATIC_LIB) $(SHARED_LIB)
	cp $(STATIC_LIB) /usr/local/lib/
	cp $(SHARED_LIB) /usr/local/lib/
	ldconfig

# Full install
install: install-headers install-libs

# Test targets
test: $(TEST_RUNNER) $(UNIT_TESTS)
	@echo "Running unit tests..."
	@./$(UNIT_TESTS)
	@echo ""
	@echo "Running integration tests..."
	@./$(TEST_RUNNER)

test-unit: $(UNIT_TESTS)
	@echo "Running unit tests..."
	@./$(UNIT_TESTS)

test-integration: $(TEST_RUNNER)
	@echo "Running integration tests..."
	@./$(TEST_RUNNER)

test-performance: $(BENCHMARK)
	@echo "Running performance benchmarks..."
	@./$(BENCHMARK)

test-all: test-unit test-integration test-performance

test-memory: $(UNIT_TESTS) $(TEST_RUNNER)
	@echo "Running memory leak tests..."
	@valgrind --leak-check=full --error-exitcode=1 ./$(UNIT_TESTS)
	@valgrind --leak-check=full --error-exitcode=1 ./$(TEST_RUNNER) --unit

# Debugging targets
debug: CC_FLAGS += -DDEBUG -O0
debug: ASM_FLAGS += -DDEBUG
debug: all

# Profiling targets
profile: CC_FLAGS += -pg
profile: all

# Memory checking
valgrind: $(TEST_RUNNER)
	valgrind --leak-check=full --track-origins=yes $(TEST_RUNNER)

# Code coverage
coverage: CC_FLAGS += --coverage
coverage: LD_FLAGS += --coverage
coverage: all
	@$(TEST_RUNNER)
	@gcov $(C_SOURCES)

# Documentation generation
docs:
	@echo "Generating documentation..."
	@doxygen docs/Doxyfile

# Clean targets
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

clean-all: clean
	rm -rf docs/html docs/latex *.gcov *.gcda *.gcno

# Example program
EXAMPLE_DIR = examples
EXAMPLE_SRC = $(EXAMPLE_DIR)/example.c
EXAMPLE_BIN = $(EXAMPLE_DIR)/example

$(EXAMPLE_BIN): $(EXAMPLE_SRC) $(STATIC_LIB)
	@echo "Building example program..."
	@mkdir -p $(EXAMPLE_DIR)
	$(CC) $(CC_FLAGS) -no-pie -I$(INCLUDE_DIR) -L$(LIB_DIR) $(EXAMPLE_SRC) -lmlasm -lm -o $@

example: $(EXAMPLE_BIN)

run-example: example
	@echo "Running example program..."
	@LD_LIBRARY_PATH=$(LIB_DIR) ./$(EXAMPLE_BIN)

# Development helpers
format:
	@echo "Formatting C code..."
	@find . -name "*.c" -o -name "*.h" | xargs clang-format -i

lint:
	@echo "Running static analysis..."
	@cppcheck --enable=all --std=c99 $(SRC_DIR)

# Print build information
info:
	@echo "Build Information:"
	@echo "  NASM: $(shell $(NASM) --version)"
	@echo "  GCC: $(shell $(CC) --version | head -1)"
	@echo "  CPU Features: $(CPU_FEATURES)"
	@echo "  ASM Flags: $(ASM_FLAGS)"
	@echo "  CC Flags: $(CC_FLAGS)"

.PHONY: all directories test-unit test-integration test-performance test-all debug profile valgrind coverage docs clean clean-all example run-example format lint install install-headers install-libs info
