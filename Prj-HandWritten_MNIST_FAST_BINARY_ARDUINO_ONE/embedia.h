/* 
 * EmbedIA 
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */
#ifndef _EMBEDIA_H
#define _EMBEDIA_H

#include <stdint.h>
#include <math.h>
#include "Arduino.h"

#define binary_block_size 8


/* definition of global masks and data types */



#if binary_block_size == 8
typedef uint8_t xBITS;
static const uint8_t mascara_global_bits[8] = { 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif binary_block_size == 16
typedef uint16_t xBITS;
static const uint16_t mascara_global_bits[16] = { 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif binary_block_size == 32
typedef uint32_t xBITS;
static const uint32_t mascara_global_bits[32] = { 2147483648 , 1073741824 , 536870912 , 268435456 , 134217728 , 67108864 , 33554432 , 16777216 , 8388608 , 4194304 , 2097152 , 1048576 , 524288 , 262144 , 131072 , 65536 , 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#elif binary_block_size == 64
typedef uint64_t xBITS;
static const uint64_t mascara_global_bits[64] = { 9223372036854775808 , 4611686018427387904 , 2305843009213693952 , 1152921504606846976 , 576460752303423488 , 288230376151711744 , 144115188075855872 , 72057594037927936 , 36028797018963968 , 18014398509481984 , 9007199254740992 , 4503599627370496 , 2251799813685248 , 1125899906842624 , 562949953421312 , 281474976710656 , 140737488355328 , 70368744177664 , 35184372088832 , 17592186044416 , 8796093022208 , 4398046511104 , 2199023255552 , 1099511627776 , 549755813888 , 274877906944 , 137438953472 , 68719476736 , 34359738368 , 17179869184 , 8589934592 , 4294967296 , 2147483648 , 1073741824 , 536870912 , 268435456 , 134217728 , 67108864 , 33554432 , 16777216 , 8388608 , 4194304 , 2097152 , 1048576 , 524288 , 262144 , 131072 , 65536 , 32768 , 16384 , 8192 , 4096 , 2048 , 1024 , 512 , 256 , 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#else
typedef uint8_t xBITS;
static const uint8_t mascara_global_bits[8] = { 128 , 64 , 32 , 16 , 8 , 4 , 2 , 1};
#endif // binary_block_size


/* STRUCTURE DEFINITION */

/*
 * Structure that models a binary neuron.
 * Specifies the weights of the neuron as a vector (xBITS  * weights) and the bias (float bias).
 */

typedef struct{
    const xBITS  * weights;
}quant_neuron_t;


/*
 * Structure that models a binary dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of quant neurons (quant_neuron_t  * neurons). 
 */

typedef struct{
    uint8_t n_neurons;
    quant_neuron_t * neurons;
}quantdense_layer_t;

typedef struct{
    uint16_t length;
    float  * data;
}data1d_t;

typedef struct{
    uint16_t length;
    uint8_t  * data;
}data1d_b_t;


/*
 * Structure that models a neuron.
 * Specifies the weights of the neuron as a vector (float * weights) and the bias (float bias).
 */
typedef struct{
    const float  * weights;
    float  bias;
}neuron_t;

/*
 * Structure that models a dense layer.
 * Specifies the number of neurons (uint16_t n_neurons) and a vector of neurons (neuron_t * neurons). 
 */
typedef struct{
    uint8_t n_neurons;
    neuron_t * neurons;
}dense_layer_t;

/*********************************************************************************************************************************/




/* 
 * Structure for BatchNormalization layer.
 * Contains vectors for the four parameters used for normalization.
 * The number of each of the parameters is determined by the number of channels of the previous layer.
 */
typedef struct {
    uint16_t length;
    const float *beta;
    // const float *gamma;
    const float *moving_mean;
    const float *moving_inv_std_dev; // = gamma / sqrt(moving_variance + epsilon)
} batch_normalization_layer_t;


/* LIBRARY FUNCTIONS PROTOTYPES */

void quantdense_layer_b(quantdense_layer_t dense_layer, data1d_b_t input, data1d_t * output);



/* 
 * dense_layer()
 * Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *   - dense_layer => structure with the weights of the neurons of the dense layer.  
 *   - input       => structure data1d_t with the input data to process. 
 *   - *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output);

     
/* 
 * argmax()
 *  Finds the index of the largest value within a vector of data (data1d_t)
 * 
 * Parameters:
 *  data => data of type data1d_t to search for max.
 *
 * Returns:
 *  search result - index of the maximum value
 */
uint16_t argmax(data1d_t data);


/***************************************************************************************************************************/
/* Activation functions/layers */

void softmax_activation(float *data, uint16_t length);

void relu_activation(float *data, uint16_t length);


//void batch_normalization_layer(batch_normalization_layer_t norm, uint32_t length, float *data);


void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data);



/* functions */
/*
* sign function, used to binarize the input
*/
static inline uint8_t sign(float x);


/*
* Brian Kernighan's algorithm, is used to count high bits (logical 1) 
* efficiently.
*/
static inline int count_set_bits_Brian_Kernighan_algorithm(xBITS n);


/*
* POPCOUNT function 
*/
static inline float POPCOUNT(xBITS n);


/*
* applies the XNOR function efficiently, between two numbers loaded in 
* registers
*/
static inline xBITS XNOR(register xBITS a,register xBITS b);

/* Tranformation Layers
 *
 */



#endif

