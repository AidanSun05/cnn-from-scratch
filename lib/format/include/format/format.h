#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

const uint8_t LayerType_Linear = 0;
const uint8_t LayerType_Conv2D = 1;
const uint8_t LayerType_MaxPool2D = 2;
const uint8_t LayerType_Flatten = 4;

const uint8_t ActivationType_None = 0;
const uint8_t ActivationType_ReLU = 1;
const uint8_t ActivationType_Softmax = 2;

typedef struct {
    char magic[4];
    uint32_t version;
    uint32_t numLayers;
    uint32_t maxBufSize;
    uint32_t padding[4];
} ModelHeader;

typedef struct {
    uint8_t type;
    uint8_t activation;
    uint16_t padding1;

    // Dimensions (use what's needed per layer)
    uint32_t dim[8];

    uint32_t weightOffset; // Byte offset to weights in file
    uint32_t biasOffset; // Byte offset to bias in file
    uint32_t weightSize; // Number of floats in weight matrix
    uint32_t biasSize; // Number of floats in bias vector
} LayerDescriptor;

#ifdef __cplusplus
}
#endif
