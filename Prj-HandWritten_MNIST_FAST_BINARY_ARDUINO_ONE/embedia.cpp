/* 
 * EmbedIA 
 * C LIBRARY FOR THE IMPLEMENTATION OF NEURAL NETWORKS ON MICROCONTROLLERS
 */

#include "embedia.h"


typedef struct{
	size_t size;
	void  * data;
} raw_buffer;


raw_buffer buffer1 = {0, NULL};
raw_buffer buffer2 = {0, NULL};

void * swap_alloc(size_t s){ 
	static raw_buffer * last_buff = &buffer2;
	last_buff = (last_buff==&buffer1) ? &buffer2 : &buffer1;
	
	if (last_buff->size < s){
		last_buff->data = realloc(last_buff->data, s);
		last_buff->size = s;
	}

	return last_buff->data;
}




static inline uint8_t sign(float x){
    if(x>=0){
        return 1;
    }else{
        return 0;
    }
}


static inline int count_set_bits_Brian_Kernighan_algorithm(xBITS n) {
  /*register int count = 0;
  while(n) {
    count++;
    n = n & (n-1);
  }

  return count;*/
  return __builtin_popcount(n);
}



static inline float POPCOUNT(xBITS n){

    return 2*count_set_bits_Brian_Kernighan_algorithm(n) - binary_block_size;

}



static inline xBITS XNOR(register xBITS a,register xBITS b){

    return ~(a^b);

}




void quantdense_layer_b(quantdense_layer_t dense_layer, data1d_b_t input, data1d_t * output){

    uint8_t bloques_enteros = input.length / binary_block_size;
    uint8_t long_ult = input.length % binary_block_size;
    output->length = dense_layer.n_neurons;
	output->data = (float*)swap_alloc(sizeof(float)*dense_layer.n_neurons);

    uint8_t i;
    register uint8_t cont;
    uint8_t j;
    xBITS tot=0;

    size_t iterador;
    for(iterador=0;iterador<output->length;iterador++){ //valores a cero
        output->data[iterador] = 0;
    }

    for(i=0;i<bloques_enteros;i++){
        tot = input.data[i];
        for(cont = 0;cont<dense_layer.n_neurons;cont++){
            output->data[cont] += POPCOUNT(XNOR(dense_layer.neurons[cont].weights[i],tot));
        }
  
    }
	if(long_ult!=0){
        tot = input.data[bloques_enteros];
        for(cont = 0;cont<dense_layer.n_neurons;cont++){
            output->data[cont] += POPCOUNT(XNOR(dense_layer.neurons[cont].weights[bloques_enteros],tot)) - (binary_block_size-long_ult);
        }
    }

}




/* 
 * neuron_forward()
 *  Function that performs the forward of a neuron before a given set of input data.
 * Parameters:
 *  - neuron_t neuron => neuron with its weights and bias loaded.
 *  - data1d_t input => input data in vector form (data1d_t).
 * Returns:
 *  - float => result of the operation.
 */
 
 static float neuron_forward(neuron_t neuron, data1d_t input){
	uint16_t i;
	float result = 0;

	for(i=0;i<input.length;i++){
		result += input.data[i]*neuron.weights[i];
	}

	return result + neuron.bias;
}

/* 
 * dense_layer()
 * Performs feed forward of a dense layer (dense_layer_t) on a given input data set.
 * Parameters:
 *   - dense_layer => structure with the weights of the neurons of the dense layer.  
 *   - input       => structure data1d_t with the input data to process. 
 *   - *output     => structure data1d_t to store the output result.
 */
void dense_layer(dense_layer_t dense_layer, data1d_t input, data1d_t * output){
	uint8_t i;

	output->length = dense_layer.n_neurons;
	output->data = (float*)swap_alloc(sizeof(float)*dense_layer.n_neurons);
	
	for(i=0;i<dense_layer.n_neurons;i++){
		output->data[i] = neuron_forward(dense_layer.neurons[i],input);
	}
}

/* 
 * softmax activation function
 * Parámeters:
 *          *data  => array of values to update
 *          length => numbers of values to update
 */
void softmax_activation(float *data, uint16_t length){
	float m = -INFINITY;
	for (uint16_t i = 0; i < length; i++) {
		if (data[i] > m) {
			m = data[i];
		}
	}

	float sum = (0.0);
	for (uint16_t i = 0; i < length; i++) {
		sum += exp(data[i] - m);
	}

	float offset = m + log(sum);
	for (uint16_t i = 0; i < length; i++) {
		data[i] = exp(data[i] - offset);
	}
}


/* 
 * relu activation function
 * Parameters:
 *          *data  => array of values to update
 *          length => numbers of values to update
 */
void relu_activation(float *data, uint16_t length){
	uint16_t i;

	for (i=0;i<(length);i++){
		data[i] = data[i] < 0 ? 0 : data[i];
	}
}


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
 uint16_t argmax(data1d_t data){
	float max = data.data[0];
	uint16_t pos = 0;

	for(uint16_t i=1;i<data.length;i++){
		if(data.data[i]>max){
			max = data.data[i];
			pos = i;
		} 
	}
	
	return pos;
}


void batch_normalization1d_layer(batch_normalization_layer_t layer, data1d_t *data) {
    uint16_t i;

    for (i = 0; i < data->length; i++) {
        data->data[i] = (data->data[i] - layer.moving_mean[i]) * layer.moving_inv_std_dev[i] + layer.beta[i];
    }
}


