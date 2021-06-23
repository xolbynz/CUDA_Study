CC = g++
CFLAGS = -g -Wall -O3 -ffast-math -flto -march=armv8.2-a
SRCS = main.cpp
PROG = main

OPENCV = `pkg-config opencv4 --libs --cflags`
LIBS = $(OPENCV)

$(PROG):$(SRCS)
	$(CC) $(CFLAGS) -o $(PROG) $(SRCS) $(LIBS)
