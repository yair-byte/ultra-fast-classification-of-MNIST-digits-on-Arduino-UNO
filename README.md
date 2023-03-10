# Ultra-fast-classification-of-MNIST-digits-on-Arduino-UNO ✍🏼 🧠 🚀

Handwritten digit classification model for Arduino UNO using a binary neural network with binarized input. TensorFlow/Keras and Larq were used for the implementation and training of the model, for the export to C code the EmbedIA library was used.

Images from the MNIST dataset were downsized to 24x24x1 and then normalized and binarized to -1 or 1.

The architecture of the model is as follows:

![model_mnist_arduino_uno h5 (1)](https://user-images.githubusercontent.com/56457143/222915228-1e98ed4c-310a-4c32-b38a-25be83ac304a.jpg)
It occupies a total of 1340 Bytes on the Arduino UNO board. The number of MACs operations is 5.86 K and it has a total of 5.89 K parameters.


Metrics:

Accuracy: 88.72% (MNIST test dataset)

Loss: 0.383 (MNIST test dataset)

Average inference time: 20 ms

Program storage space: 8540 Bytes (26%)

Global variables: 1778 Bytes (86% of dynamic memory)


Confusion Matrix:

<img width="400" alt="image" src="https://user-images.githubusercontent.com/56457143/222914784-43a070f0-cfab-41cb-be64-663b1d98c41c.png">

Some predictions:

<img width="600" alt="image" src="https://user-images.githubusercontent.com/56457143/222914828-a7d90437-78d4-4e9b-8f4b-d19d536abbb7.png">

To use the program just open the .ino file and upload it to your Arduino Uno, enjoy!

