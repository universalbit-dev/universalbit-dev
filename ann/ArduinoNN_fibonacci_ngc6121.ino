// ******************************************************************
// ArduinoANN - An artificial neural network for Arduino
// This section introduces the code as an ANN implementation for Arduino.
// Reference: robotics.hobbizine.com/arduinoann.html
// ******************************************************************

// Including math.h for mathematical operations.
#include <math.h>

// ******************************************************************
// Network Configuration - Customized per network
// Defines the structure and parameters of the neural network.
// ******************************************************************

// Number of training patterns
const int PatternCount = 10;

// Number of nodes in the input layer
const int InputNodes = 7;

// Number of nodes in the hidden layer
const int HiddenNodes = 8;

// Number of nodes in the output layer
const int OutputNodes = 4;

// Learning rate controls how much the weights are updated during training
const float LearningRate = 0.3;

// Momentum helps smooth the weight updates and avoid oscillations
const float Momentum = 0.9;

// Maximum initial weight value for random initialization
const float InitialWeightMax = 0.5;

// Training success threshold (error rate)
const float Success = 0.0004;

// Input data: Fibonacci sequences
// Each row represents a training pattern with 7 input values.
const byte Input[PatternCount][InputNodes] = {
  { 0, 1, 1, 2, 3, 5, 8 }, 
  { 1, 1, 2, 3, 5, 8, 13 },
  { 1, 2, 3, 5, 8, 13, 21 },
  { 2, 3, 5, 8, 13, 21, 34 },
  { 3, 5, 8, 13, 21, 34, 55 }, 
  { 5, 8, 13, 21, 34, 55, 89 }, 
  { 8, 13, 21, 34, 55, 89, 144 }, 
  { 13, 21, 34, 55, 89, 144, 233 }, 
  { 21, 34, 55, 89, 144, 233, 377 }, 
  { 34, 55, 89, 144, 233, 377, 610 } 
};

// Target data: Hypothetical binary output values
// Each row represents the desired output for the corresponding input pattern.
const byte Target[PatternCount][OutputNodes] = {
  { 1, 0, 1, 0 },  
  { 1, 0, 1, 1 }, 
  { 1, 1, 0, 0 }, 
  { 1, 1, 0, 1 }, 
  { 0, 1, 1, 0 }, 
  { 0, 1, 1, 1 }, 
  { 0, 0, 1, 0 }, 
  { 0, 0, 1, 1 }, 
  { 1, 0, 0, 0 }, 
  { 1, 0, 0, 1 }
};

// ******************************************************************
// Initialization Variables
// ******************************************************************

int i, j, p, q, r; // Indices and counters
int ReportEvery1000; // Counter for progress reporting
int RandomizedIndex[PatternCount]; // Array for randomizing training patterns
long TrainingCycle; // Counter for the number of training cycles
float Rando; // Random number for weight initialization
float Error; // Cumulative error during training
float Accum; // Accumulator for summing weighted inputs

// Arrays for storing activations and weights
float Hidden[HiddenNodes]; // Activations for hidden layer
float Output[OutputNodes]; // Activations for output layer
float HiddenWeights[InputNodes+1][HiddenNodes]; // Weights between input and hidden layers
float OutputWeights[HiddenNodes+1][OutputNodes]; // Weights between hidden and output layers
float HiddenDelta[HiddenNodes]; // Error deltas for hidden layer
float OutputDelta[OutputNodes]; // Error deltas for output layer
float ChangeHiddenWeights[InputNodes+1][HiddenNodes]; // Weight change history for input-hidden weights
float ChangeOutputWeights[HiddenNodes+1][OutputNodes]; // Weight change history for hidden-output weights

// ******************************************************************
// Arduino Setup Function
// ******************************************************************
void setup(){
  Serial.begin(9600); // Initialize serial communication
  randomSeed(analogRead(3)); // Seed random number generator
  ReportEvery1000 = 1; // Initialize report counter
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p; // Initialize randomized indices
  }
}  

// ******************************************************************
// Arduino Loop Function
// ******************************************************************
void loop (){

  // Initialize weights randomly
  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0;
      Rando = float(random(100))/100;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  for( i = 0 ; i < OutputNodes ; i++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0;  
      Rando = float(random(100))/100;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  Serial.println("Initial/Untrained Outputs: ");
  toTerminal(); // Display initial outputs

  // Start training loop
  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    

    // Randomize order of training patterns
    for( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p]; 
      RandomizedIndex[p] = RandomizedIndex[q]; 
      RandomizedIndex[q] = r;
    }

    Error = 0.0; // Reset error for this cycle

    // Iterate through training patterns
    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

      // Compute hidden layer activations
      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i];
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i];
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)); // Sigmoid activation
      }

      // Compute output layer activations and errors
      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i];
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i];
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)); // Sigmoid activation
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]);   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]);
      }

      // Backpropagate errors to hidden layer
      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j];
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]);
      }

      // Update weights
      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i];
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i];
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i];
        }
      }

      for( i = 0 ; i < OutputNodes ; i++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i];
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i];
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i];
          OutputWeights[j][i] += ChangeOutputWeights[j][i];
        }
      }
    }

    // Display progress every 1000 cycles
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0) {
      Serial.println(); 
      Serial.println(); 
      Serial.print ("TrainingCycle: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.println (Error, 5);

      toTerminal(); // Display current outputs

      if (TrainingCycle == 1) {
        ReportEvery1000 = 999;
      } else {
        ReportEvery1000 = 1000;
      }
    }    

    // Stop training if error is below threshold
    if( Error > Success ) break;  
  }

  Serial.println();
  Serial.println(); 
  Serial.print ("TrainingCycle: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.println (Error, 5);

  toTerminal(); // Display final outputs

  Serial.println();  
  Serial.println();
  Serial.println("Training Set Solved!");
  Serial.println("--------"); 
  Serial.println();
  Serial.println();  
  ReportEvery1000 = 0;
}

// ******************************************************************
// toTerminal Function
// Displays input, target, and output for each training pattern.
// ******************************************************************
void toTerminal(){
  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("  Training Pattern: ");
    Serial.println (p);      
    Serial.print ("  Input ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("  Target ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }

    // Compute hidden and output layer activations
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i];
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i];
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum));
    }

    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i];
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i];
      }
      Output[i] = 1.0/(1.0 + exp(-Accum));
    }

    // Display output
    Serial.print ("  Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }
}
