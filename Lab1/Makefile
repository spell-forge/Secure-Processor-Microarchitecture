CC = gcc
CFLAGS = -Wall -Wextra -O2

TARGET = clefia
SRC = q1.c

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -D_CLEFIA_TEST -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o