# Compiler and Flags
CXX      = clang++
CXXFLAGS = -std=c++17 -Xpreprocessor -fopenmp
TARGET   = manifold
SRC      = manifold.cpp manifold_operators.cpp database.cpp

# Paths (Using the specific Homebrew paths from your command)
OMP_PATH = /usr/local/Cellar/libomp/22.1.4
INCLUDES = -I$(OMP_PATH)/include
LDFLAGS  = -L$(OMP_PATH)/lib -lomp

# Default rule, the manifold can be created running executions of all elements in the array in parallel.
all: $(TARGET)

# Link the executable
$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $(SRC) -o $(TARGET) $(LDFLAGS)

# Clean up build files
clean:
	rm -f $(TARGET)

# Run the program
run: all
	./$(TARGET)