#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define FILE_TRAIN_IMAGE "train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels-idx1-ubyte"
#define LENET_FILE       "model.dat"
#define COUNT_TRAIN      60000
#define COUNT_TEST       10000

double wtime(void) {
    struct timeval etstart;
    if (gettimeofday(&etstart, NULL) == -1)
        perror("Error: calling gettimeofday() not successful.\n");
    return ((etstart.tv_sec) * 1000 + etstart.tv_usec / 1000.0);
}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[]) {
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image || !fp_label) return 1;

    fseek(fp_image, 16, SEEK_SET);
    fseek(fp_label, 8, SEEK_SET);
    fread(data, sizeof(*data) * count, 1, fp_image);
    fread(label, count, 1, fp_label);
    fclose(fp_image);
    fclose(fp_label);
    return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size) {
        TrainBatch(lenet, train_data + i, train_label + i, batch_size);
        if (rank == 0 && i * 100 / total_size > percent)
            printf("batchsize:%d\ttrain:%2d%%\n", batch_size, percent = i * 100 / total_size);
    }
}

int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size) {
    int correct = PredictBatch(lenet, test_data, test_label, total_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Total correct predictions: %d\n", correct);
        printf("Accuracy: %.2f%%\n", (correct * 100.0) / total_size);
    }

    return correct;
}

void foo() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
    uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
    image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
    uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));

    if (rank == 0) {
        if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL)) {
            printf("ERROR: Dataset not found!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL)) {
            printf("ERROR: Dataset not found!\n");
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }

    // Broadcast datasets
    MPI_Bcast(train_data, COUNT_TRAIN * sizeof(image), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(train_label, COUNT_TRAIN, MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_data, COUNT_TEST * sizeof(image), MPI_BYTE, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_label, COUNT_TEST, MPI_BYTE, 0, MPI_COMM_WORLD);

    LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
    if (rank == 0) {
        if (load(lenet, LENET_FILE)) {
            Initial(lenet);
        }
    }

    // Broadcast the model to all processes
    MPI_Bcast(lenet, sizeof(LeNet5), MPI_BYTE, 0, MPI_COMM_WORLD);

    clock_t start = clock();
    double bef_train = wtime();

    int batches[] = { 300 };
    for (int i = 0; i < sizeof(batches) / sizeof(*batches); ++i) {
        training(lenet, train_data, train_label, batches[i], COUNT_TRAIN);
    }

    clock_t end_train = clock();
    double aft_train = wtime();

    testing(lenet, test_data, test_label, COUNT_TEST);

    clock_t end_test = clock();
    double aft_test = wtime();

    if (rank == 0) {
        printf("Training CPU time: %u\n", (unsigned)(end_train - start));
        printf("Training CPU time in seconds: %fs\n", (float)(end_train - start) / CLOCKS_PER_SEC);
        printf("Testing CPU time: %u\n", (unsigned)(end_test - end_train));
        printf("Testing CPU time in seconds: %fs\n", (float)(end_test - end_train) / CLOCKS_PER_SEC);
        printf("Training time: %lfs\n", (aft_train - bef_train) / 1000.0);
        printf("Testing time: %lfs\n", (aft_test - aft_train) / 1000.0);
    }

    free(lenet);
    free(train_data);
    free(train_label);
    free(test_data);
    free(test_label);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    foo();

    MPI_Finalize();
    return 0;
}
