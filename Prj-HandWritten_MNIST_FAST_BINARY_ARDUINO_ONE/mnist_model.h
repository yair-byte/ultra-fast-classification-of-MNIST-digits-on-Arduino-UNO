/* EmbedIA model */
#ifndef MNIST_MODEL_H
#define MNIST_MODEL_H

#include "embedia.h"

#define INPUT_SIZE 576


void model_init();

void model_predict(data1d_b_t input, data1d_t * output);

int model_predict_class(data1d_b_t input, data1d_t * results);

#endif
