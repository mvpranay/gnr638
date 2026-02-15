CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Iinclude
SRCDIR = src
OBJDIR = obj
INCDIR = include
TARGET = main

all: $(OBJDIR) $(TARGET)

$(OBJDIR):
	mkdir -p $(OBJDIR)

$(TARGET): $(OBJDIR)/main.o $(OBJDIR)/tensor.o $(OBJDIR)/ops.o $(OBJDIR)/linear.o $(OBJDIR)/init.o $(OBJDIR)/optimizer.o $(OBJDIR)/loss.o
	$(CXX) $(CXXFLAGS) -o $(TARGET) $^

$(OBJDIR)/main.o: main.cpp $(INCDIR)/linear.hpp $(INCDIR)/optimizer.hpp $(INCDIR)/loss.hpp $(INCDIR)/tensor.hpp $(INCDIR)/ops.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/tensor.o: $(SRCDIR)/tensor.cpp $(INCDIR)/tensor.hpp $(INCDIR)/ops.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/ops.o: $(SRCDIR)/ops.cpp $(INCDIR)/ops.hpp $(INCDIR)/tensor.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/linear.o: $(SRCDIR)/linear.cpp $(INCDIR)/linear.hpp $(INCDIR)/tensor.hpp $(INCDIR)/ops.hpp $(INCDIR)/init.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/init.o: $(SRCDIR)/init.cpp $(INCDIR)/init.hpp $(INCDIR)/tensor.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/optimizer.o: $(SRCDIR)/optimizer.cpp $(INCDIR)/optimizer.hpp $(INCDIR)/tensor.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJDIR)/loss.o: $(SRCDIR)/loss.cpp $(INCDIR)/loss.hpp $(INCDIR)/tensor.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR) $(TARGET)

build:
	python3 setup.py build_ext --inplace

rmbuild:
	rm -rf build APDNN.cpython-312-x86_64-linux-gnu.so

.PHONY: all clean rmbuild build