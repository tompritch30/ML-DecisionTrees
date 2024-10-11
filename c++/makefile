# Compiler
CXX = g++

# Compiler flags
CXXFLAGS = -Wall -fPIC -shared -pthread

# Source files
SOURCES = optimisation.cpp tree.cpp forest.cpp

# Output library
OUTPUT = liboptimisation.so

# Default target
all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CXX) $(CXXFLAGS) -o $@ $(SOURCES)

# Clean rule for cleaning up
clean:
	rm -f $(OUTPUT)

.PHONY: all clean
