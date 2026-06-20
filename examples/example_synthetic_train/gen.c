#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>

#define W 32
#define H 32
#define C 3

#define NUM_CLASSES 3

static inline int idx(int n, int y, int x, int c) {
    return ((n * H + y) * W + x) * C + c;
}

void setPixel(float* data, int n, int x, int y, float r, float g, float b) {
    if (x < 0 || x >= W || y < 0 || y >= H) return;

    data[idx(n, y, x, 0)] = r;
    data[idx(n, y, x, 1)] = g;
    data[idx(n, y, x, 2)] = b;
}

void drawSquare(float* data, int n, int cx, int cy, int size, float r, float g, float b) {
    int half = size / 2;

    for (int y = cy - half; y <= cy + half; y++) {
        for (int x = cx - half; x <= cx + half; x++) {
            setPixel(data, n, x, y, r, g, b);
        }
    }
}

void drawCircle(float* data, int n, int cx, int cy, int radius, float r, float g, float b) {
    for (int y = cy - radius; y <= cy + radius; y++) {
        for (int x = cx - radius; x <= cx + radius; x++) {
            int dx = x - cx;
            int dy = y - cy;

            if (dx * dx + dy * dy <= radius * radius) {
                setPixel(data, n, x, y, r, g, b);
            }
        }
    }
}

void drawTriangle(float* data, int n, int cx, int cy, int size, float r, float g, float b) {
    for (int row = 0; row < size; row++) {
        int start = cx - row;
        int end = cx + row;

        int y = cy + row;

        for (int x = start; x <= end; x++) {
            setPixel(data, n, x, y, r, g, b);
        }
    }
}

void writeArray(const char* fname, void* data, size_t size, size_t length) {
    FILE* f = fopen(fname, "wb");
    if (f == NULL) {
        printf("Error opening file");
        return;
    }

    fwrite(data, size, length, f);
    fclose(f);
}

int generateDataset(const char* dirName, const char* dataName, const char* labelsName, int N) {
    srand((unsigned)time(NULL));

    float* images = calloc(N * H * W * C, sizeof(float));
    uint8_t* labels = malloc(sizeof(uint8_t) * N);

    if (mkdir(dirName, 0777) != 0 && errno != EEXIST) {
        perror("Error creating images directory\n");
        return 1;
    }

    for (int n = 0; n < N; n++) {
        uint8_t cls = (uint8_t)(rand() % NUM_CLASSES);
        labels[n] = cls;

        int cx = 8 + rand() % 16;
        int cy = 8 + rand() % 16;

        int size = 4 + rand() % 8;

        float r = (float)rand() / RAND_MAX;
        float g = (float)rand() / RAND_MAX;
        float b = (float)rand() / RAND_MAX;

        switch (cls) {
            case 0:
                drawSquare(images, n, cx, cy, size, r, g, b);
                break;

            case 1:
                drawCircle(images, n, cx, cy, size, r, g, b);
                break;

            case 2:
                drawTriangle(images, n, cx, cy, size, r, g, b);
                break;
        }

        char fname[30] = { 0 };
        snprintf(fname, sizeof(fname), "%s/%d.ppm", dirName, n);
        FILE* f = fopen(fname, "wb");
        if (f == NULL) {
            printf("Error opening file\n");
            break;
        }

        const char* header = "P6\n32 32\n255\n";
        fwrite(header, sizeof(char), strlen(header), f);
        for (int i = 0; i < H; i++) {
            for (int j = 0; j < W; j++) {
                for (int k = 0; k < C; k++) {
                    float val = images[idx(n, i, j, k)];
                    uint8_t p = (uint8_t)(val * 255.0f);
                    fwrite((char*)&p, 1, 1, f);
                }
            }
        }
        fclose(f);
    }

    printf("Generated dataset '%s', '%s'.\n", dataName, labelsName);
    printf("Tensor shape: [%d, %d, %d, %d]\n", N, H, W, C);

    writeArray(dataName, images, sizeof(float), N * H * W * C);
    writeArray(labelsName, labels, sizeof(uint8_t), N);

    free(images);
    free(labels);

    return 0;
}

int main() {
    if (generateDataset("images-train", "data-train.dat", "labels-train.dat", 20000) != 0) {
        return 1;
    }

    if (generateDataset("images-test", "data-test.dat", "labels-test.dat", 5000) != 0) {
        return 1;
    }

    return 0;
}
