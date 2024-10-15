CC = gcc
CFLAGS = -Wall -lm -fopenmp 
TARGET1 = fft_vect
TARGET2 = fft_omp_first
TARGET3 = fft
all: $(TARGET1) $(TARGET2) $(TARGET3)

$(TARGET1): fft_vect.c
	$(CC) $^ -o $@ $(CFLAGS) -mavx2

$(TARGET2): fft_omp_first.c
	$(CC) $^ -o $@ $(CFLAGS)

$(TARGET3): fft.c
	$(CC) $^ -o $@ -lm

clean:
	rm -f $(TARGET1) $(TARGET2)
