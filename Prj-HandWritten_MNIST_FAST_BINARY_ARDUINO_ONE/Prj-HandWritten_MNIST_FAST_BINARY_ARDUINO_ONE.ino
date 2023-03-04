#include "Arduino.h"
#include "embedia.h"
#include "mnist_model.h"
#include "example_file.h"

data1d_b_t input = { INPUT_SIZE,  NULL};


data1d_t results;


void setup(){

    // Serial inicialization
    Serial.begin(9600);

    // Model initialization
    model_init();

}

void loop(){


    unsigned long inicio = millis(); // Guarda el tiempo actual

    
    // sample intitialization
    input.data = sample_data;

    model_init();

    // make model prediction
    // uncomment corresponding code
    
    int prediction = model_predict_class(input, &results);

    unsigned long fin = millis(); // Guarda el tiempo actual nuevamente
    unsigned long tiempo_transcurrido = fin - inicio; // Calcula la duración en milisegundos
    Serial.print("Tiempo transcurrido: ");
    Serial.print(tiempo_transcurrido);
    Serial.println(" milisegundos");

    // print predicted class id
    Serial.print("Prediction class id: ");
    Serial.println(prediction);

    Serial.print("confianza: ");
    Serial.println(results.data[prediction]);

    delay(5000);



}