#include "inference/nn.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core/conv.h"
#include "core/gemm.h"
#include "core/ops.h"
#include "format/format.h"

struct NNModelInternal {
    size_t batchSize;
    uint32_t numLayers;
    LayerDescriptor* layers;
    char* weights;
    float* outBufs[2];
};

void nnInit(void) {
    gemmInit();
}

void nnCleanup(void) {
    gemmCleanup();
}

int nnLoad(const char* path, NNModel** model, size_t batchSize) {
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        return NN_FILE_ERROR;
    }

    ModelHeader header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return NN_HEADER_READ_ERROR;
    }

    if (memcmp(header.magic, "NNML", 4) != 0) {
        fclose(f);
        return NN_MAGIC_NUMBER_MISMATCH;
    }

    if (header.version != 1) {
        fclose(f);
        return NN_VERSION_MISMATCH;
    }

    // Allocate model
    struct NNModelInternal* modelTmp = malloc(sizeof(struct NNModelInternal));
    modelTmp->batchSize = batchSize;
    modelTmp->numLayers = header.numLayers;
    modelTmp->layers = malloc(sizeof(LayerDescriptor) * header.numLayers);

    // Read all descriptors
    if (fread(modelTmp->layers, sizeof(LayerDescriptor), header.numLayers, f) != header.numLayers) {
        nnFree(modelTmp);
        fclose(f);
        return NN_DESCRIPTOR_READ_ERROR;
    }

    // Read all weights into a single buffer
    if (fseek(f, 0, SEEK_END) != 0) {
        nnFree(modelTmp);
        fclose(f);
        return NN_WEIGHT_READ_ERROR;
    }

    const size_t fileSize = ftell(f);
    const size_t off = sizeof(header) + sizeof(LayerDescriptor) * header.numLayers;
    const size_t weightsSize = fileSize - off;

    modelTmp->weights = aligned_alloc(32, weightsSize); // 32-byte align for AVX
    if ((fseek(f, off, SEEK_SET) != 0) || (fread(modelTmp->weights, 1, weightsSize, f) != weightsSize)) {
        nnFree(modelTmp);
        fclose(f);
        return NN_WEIGHT_READ_ERROR;
    }

    // Set up ping-pong buffers
    modelTmp->outBufs[0] = aligned_alloc(32, batchSize * header.maxBufSize * sizeof(float));
    modelTmp->outBufs[1] = aligned_alloc(32, batchSize * header.maxBufSize * sizeof(float));

    fclose(f);
    *model = modelTmp;
    return NN_OK;
}

const float* nnPredict(NNModel* model, const float* in, size_t inDims, size_t* outDims) {
    const size_t batchSize = model->batchSize;

    float** bufs = model->outBufs;
    memcpy(bufs[0], in, batchSize * inDims * sizeof(float));

    uint8_t bufIdx = 0;
    int outRows = 0;
    int outCols = 0;
    for (int i = 0; i < model->numLayers; i++) {
        const float* inBuf = bufs[bufIdx];
        float* outBuf = bufs[bufIdx ^ 1];

        const LayerDescriptor* layer = model->layers + i;
        const float* W = (float*)(model->weights + layer->weightOffset);
        const float* b = (float*)(model->weights + layer->biasOffset);

        const uint32_t* params = layer->dim;

        if (layer->type == LayerType_Conv2D) {
            const int h = params[0];
            const int w = params[1];
            const int c = params[2];
            const int k = params[3];
            const int kh = params[4];
            const int kw = params[5];
            const int outH = params[6];
            const int outW = params[7];

            outRows = batchSize * outH * outW;
            outCols = k;

            conv2dPacked(inBuf, W, b, outBuf, outH, outW, batchSize, h, w, c, k, kh, kw);
        } else if (layer->type == LayerType_Linear) {
            const int m = params[0];
            const int n = params[1];

            outRows = batchSize;
            outCols = n;

            multiply(inBuf, W, outBuf, batchSize, m, n, false, false, m, n);
            rvAdd(outBuf, b, batchSize, n);
        } else if (layer->type == LayerType_MaxPool2D) {
            const int h = params[0];
            const int w = params[1];
            const int c = params[2];
            const int kh = params[3];
            const int kw = params[4];
            const int outH = params[5];
            const int outW = params[6];

            outRows = batchSize * outH * outW;
            outCols = c;

            maxPool2dFwd(inBuf, outBuf, batchSize, h, w, c, kh, kw);
        } else if (layer->type == LayerType_Flatten) {
            continue;
        }

        if (layer->activation == ActivationType_ReLU) {
            relu(outBuf, outRows, outCols);
        } else if (layer->activation == ActivationType_Softmax) {
            softmax(outBuf, outRows, outCols);
        }

        bufIdx ^= 1;
    }

    if (outDims != NULL) {
        *outDims = outRows * outCols;
    }
    return bufs[bufIdx];
}

void nnPredictedClasses(NNModel* model, const float* pred, int numClasses, struct Prediction* out) {
    const size_t batchSize = model->batchSize;

    for (size_t i = 0; i < batchSize; i++) {
        const float* row = pred + i * numClasses;

        out[i].cls = argmax(row, numClasses);
        out[i].confidence = row[out[i].cls];
    }
}

void nnFree(NNModel* model) {
    if (model == NULL) {
        return;
    }

    free(model->layers);
    free(model->weights);
    free(model->outBufs[0]);
    free(model->outBufs[1]);

    free(model);
}
