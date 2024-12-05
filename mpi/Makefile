CC = mpicc
CFLAGS = -Wall -O3
LDFLAGS = -lm
TARGET = main

all: $(TARGET)

$(TARGET): main.o lenet.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

main.o: main.c lenet.h
	$(CC) $(CFLAGS) -c main.c

lenet.o: lenet.c lenet.h
	$(CC) $(CFLAGS) -c lenet.c

clean:
	rm -f *.o $(TARGET)

.PHONY: all clean

