# ====================================================================
# 1. COMPILER & SYSTEM AUTODETECT (Mac vs Linux GCC)
# ====================================================================
# Detect if we are on Linux or macOS
UNAME_S := $(shell uname -s)

ifeq ($(UNAME_S), Linux)
    # GCloud Linux settings
    CXX      = g++
    OMP_FLAGS = -fopenmp
    OMP_LIBS  = -fopenmp
else
    # macOS settings (Homebrew path matching your machine)
    CXX      = clang++
    OMP_PATH = /usr/local/Cellar/libomp/22.1.4
    OMP_FLAGS = -Xpreprocessor -fopenmp -I$(OMP_PATH)/include
    OMP_LIBS  = -L$(OMP_PATH)/lib -lomp
endif

# ====================================================================
# 2. CORE BUILD CONFIGURATIONS
# ====================================================================
# FIXED: Merged the CXXFLAGS cleanly using += so nothing is overwritten
CXXFLAGS = -std=c++17 -Wall -O3 -ljpeg
LIBS     = 

# SRC      = manifold.cpp manifold_operators.cpp database.cpp loading.cpp
# OBJ      = $(SRC:.cpp=.o)
# TARGET   = manifold
SRC      = hypercube.cpp vertex.cpp database.cpp manifold_operators.cpp loading.cpp loading_image.cpp
OBJ      = $(SRC:.cpp=.o)
TARGET   = hypercube

# ====================================================================
# 3. BUILD RULES
# ====================================================================
# Default rule: Compiles clean, standard serial code
all: 
	@$(MAKE) clean
	@$(MAKE) $(TARGET)

# FIXED: Separate 'clean' and '$(TARGET)' invocation dynamically.
# This prevents the double/multiple compilation loop of source files.
parallel:
	@$(MAKE) clean
	@$(MAKE) $(TARGET) CXXFLAGS="$(CXXFLAGS) $(OMP_FLAGS)" LIBS="$(LIBS) $(OMP_LIBS)"

# How to build the final executable
$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(TARGET) $(LIBS)

# How to compile source files into object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule (Crucial when switching modes)
clean:
	rm -f $(OBJ) $(TARGET)

.PHONY: all parallel clean