CC = g++
CFLAGS = -g -Wall
SRCS = cuda_test.cpp
PROG = cuda_test

OPENCV = `pkg-config opencv4 --libs --cflags`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
   $(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)