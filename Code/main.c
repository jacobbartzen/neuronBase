#include "neuralNetwork.h"

//Data Parameters
#define INPUT_SIZE 8                //Number of Different Inputs / Parameters
#define TRAINING_SIZE 15000            //How many data points to use for training

int main() {

    //Load Data from CSV File
    dataSet *data = loadCSV("housing.csv", 8, INPUT_SIZE, TRAINING_SIZE);

    //Architecture
    int neuronLayers[] = {200, 20, 1};    //Array of Neuron Counts for Each Layer
    int layers = 3;

    //Create all variables for network on heap, normalize data, and initialize weights and biases
    Network *net = createNetwork(neuronLayers, data, layers);

    // -------- Variables ----- 
    //Declare any neededed. All variables have default values set in createNetwork function, but can be changed here. More features and variables can be found in createNetwork for more advanced control.
    net->EPOCHS = 100;                  //Amount of Times to Go Through Entire Dataset
    net->LEARNING_RATE = 0.2;            //How Fast Weights change based on Error
    net->PRINT_INTERVAL = 1;           //How Often to Print Results (in Epochs)

    //Optimizer
    net->optimizer = 'N';
    //A = Adam Optimizer | Optimal Learning Rate = 0.0003
    //R = RMSProp        | Optimal Learning Rate = 0.00005
    //M = Momentum       | Optimal Learning Rate = 0.5
    //N = None           | Optimal Learning Rate = 0.2

    //Activation Function
    net->activationFunction = 'L';
    //L = Leaky ReLU
    //R = ReLU
    //S = Sigmoid

    //Load Weights - Uncomment to load weights
    //if (loadWeights(net, "networkWeights.bin")) printf("Weights loaded successfully.\n");

    //Train Network
    trainNetwork(net, data);

    //Save Weights - Uncomment to save current weights
    //if (saveWeights(net, "networkWeights.bin")) printf("Weights saved successfully.\n");

    //Test Network
    testNetwork(net, data);

    //Try Predicting an Output - change test to be anything
    float test[] = {0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5};
    float prediction = predictOutput(net, test, data);

    //Free memory when program is done
    freeMemory(net);

    freeDataSet(data);

    return 0;
}
