#pragma once

#include <stddef.h>

#define NN_OK 0
#define NN_FILE_ERROR -1
#define NN_HEADER_READ_ERROR -2
#define NN_MAGIC_NUMBER_MISMATCH -3
#define NN_VERSION_MISMATCH -4
#define NN_DESCRIPTOR_READ_ERROR -5
#define NN_WEIGHT_READ_ERROR -6

#ifdef __cplusplus
extern "C" {
#endif

typedef struct NNModelInternal NNModel;

struct Prediction {
    int cls;
    float confidence;
};

// Initializes resources needed for inference.
void nnInit(void);

// Frees resources initialized by nnInit.
void nnCleanup(void);

// Loads a model from a given path. The model will be able to make batchSize number of inferences
// with a single call to nnPredict.
int nnLoad(const char* path, NNModel** model, size_t batchSize);

// Runs the forward pass on the model and returns a matrix where row n corresponds to the prediction
// on the nth sample in the input tensor.
//
// This function writes to outDims to let the caller know the size of the returned output buffer.
// If this size is already known in advance, outDims may be set to NULL.
//
// The returned buffer is owned by the model instance and should not be freed manually.
const float* nnPredict(NNModel* model, const float* in, size_t inDims, size_t* outDims);

// Given a model and prediction from nnPredict, writes to an array of Prediction structs containing
// the class number and confidence of each prediction.
void nnPredictedClasses(NNModel* model, const float* pred, int numClasses, struct Prediction* out);

// Deallocates a model.
void nnFree(NNModel* model);

#ifdef __cplusplus
}
#endif
