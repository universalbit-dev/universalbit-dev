/******************************************************************
 * ArduinoNNFIB - An artificial neural network for the Arduino
 * Designed to learn and classify sequential Fibonacci patterns.
 ******************************************************************/

#include <math.h>
#include <Arduino.h>

/******************************************************************
 * Network Configuration
 ******************************************************************/

const int PatternCount = 10;
const int InputNodes = 7;
const int HiddenNodes = 8;
const int OutputNodes = 4;
const float LearningRate = 0.3;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0004;

// 7-step Fibonacci sequence sliding windows
// Note: Values > 255 wrap around automatically due to the 'byte' type constraints:
// - 377 wraps to 121 (377 - 256)
// - 610 wraps to 98  (610 - 512)
const byte Input[PatternCount][InputNodes] = {
  { 0, 1, 1, 2, 3, 5, 8 },          // Pattern 0
  { 1, 1, 2, 3, 5, 8, 13 },         // Pattern 1
  { 1, 2, 3, 5, 8, 13, 21 },        // Pattern 2
  { 2, 3, 5, 8, 13, 21, 34 },       // Pattern 3
  { 3, 5, 8, 13, 21, 34, 55 },      // Pattern 4
  { 5, 8, 13, 21, 34, 55, 89 },      // Pattern 5
  { 8, 13, 21, 34, 55, 89, 144 },    // Pattern 6
  { 13, 21, 34, 55, 89, 144, 233 },  // Pattern 7
  { 21, 34, 55, 89, 144, 233, 121 }, // Pattern 8 (377 % 256)
  { 34, 55, 89, 144, 233, 121, 98 }  // Pattern 9 (610 % 256)
}; 

// Target classifications matching your terminal output exactly
const byte Target[PatternCount][OutputNodes] = {
  { 1, 0, 1, 0 },  // Pattern 0
  { 1, 0, 1, 1 },  // Pattern 1
  { 1, 1, 0, 0 },  // Pattern 2
  { 1, 1, 0, 1 },  // Pattern 3
  { 0, 1, 1, 0 },  // Pattern 4
  { 0, 1, 1, 1 },  // Pattern 5
  { 0, 0, 1, 0 },  // Pattern 6
  { 0, 0, 1, 1 },  // Pattern 7
  { 1, 0, 0, 0 },  // Pattern 8
  { 1, 0, 0, 1 }   // Pattern 9
};

int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long TrainingCycle;
float Rando;
float Error;
float Accum;

float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes+1][HiddenNodes];
float OutputWeights[HiddenNodes+1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes+1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes+1][OutputNodes];

void toTerminal();

void setup(){
  Serial.begin(115200);
  while(!Serial);
  randomSeed(analogRead(3));
  ReportEvery1000 = 1;
  for( p = 0 ; p < PatternCount ; p++ ) {    
    RandomizedIndex[p] = p ;
  }
}  

void loop (){
  // Initialize HiddenWeights and ChangeHiddenWeights 
  for( i = 0 ; i < HiddenNodes ; i++ ) {    
    for( j = 0 ; j <= InputNodes ; j++ ) { 
      ChangeHiddenWeights[j][i] = 0.0 ;
      Rando = float(random(100))/100.0;
      HiddenWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }

  // Initialize OutputWeights and ChangeOutputWeights
  for( i = 0 ; i < OutputNodes ; i ++ ) {    
    for( j = 0 ; j <= HiddenNodes ; j++ ) {
      ChangeOutputWeights[j][i] = 0.0 ;  
      Rando = float(random(100))/100.0;        
      OutputWeights[j][i] = 2.0 * ( Rando - 0.5 ) * InitialWeightMax ;
    }
  }
  
  Serial.println("Initial/Untrained Outputs: ");
  toTerminal();

  // Begin training 
  for( TrainingCycle = 1 ; TrainingCycle < 2147483647 ; TrainingCycle++) {    

    // Randomize order of training patterns
    for( p = 0 ; p < PatternCount ; p++) {
      q = random(PatternCount);
      r = RandomizedIndex[p] ; 
      RandomizedIndex[p] = RandomizedIndex[q] ; 
      RandomizedIndex[q] = r ;
    }
    
    Error = 0.0 ;

    // Cycle through each training pattern
    for( q = 0 ; q < PatternCount ; q++ ) {    
      p = RandomizedIndex[q];

      // Compute hidden layer activations
      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = HiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) {
          Accum += Input[p][j] * HiddenWeights[j][i] ;
        }
        Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
      }

      // Compute output layer activations and calculate errors
      for( i = 0 ; i < OutputNodes ; i++ ) {    
        Accum = OutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          Accum += Hidden[j] * OutputWeights[j][i] ;
        }
        Output[i] = 1.0/(1.0 + exp(-Accum)) ;   
        OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i] * (1.0 - Output[i]) ;   
        Error += 0.5 * (Target[p][i] - Output[i]) * (Target[p][i] - Output[i]) ;
      }

      // Backpropagate errors to hidden layer
      for( i = 0 ; i < HiddenNodes ; i++ ) {    
        Accum = 0.0 ;
        for( j = 0 ; j < OutputNodes ; j++ ) {
          Accum += OutputWeights[i][j] * OutputDelta[j] ;
        }
        HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]) ;
      }

      // Update Input-->Hidden Weights
      for( i = 0 ; i < HiddenNodes ; i++ ) {     
        ChangeHiddenWeights[InputNodes][i] = LearningRate * HiddenDelta[i] + Momentum * ChangeHiddenWeights[InputNodes][i] ;
        HiddenWeights[InputNodes][i] += ChangeHiddenWeights[InputNodes][i] ;
        for( j = 0 ; j < InputNodes ; j++ ) { 
          ChangeHiddenWeights[j][i] = LearningRate * Input[p][j] * HiddenDelta[i] + Momentum * ChangeHiddenWeights[j][i];
          HiddenWeights[j][i] += ChangeHiddenWeights[j][i] ;
        }
      }

      // Update Hidden-->Output Weights
      for( i = 0 ; i < OutputNodes ; i ++ ) {    
        ChangeOutputWeights[HiddenNodes][i] = LearningRate * OutputDelta[i] + Momentum * ChangeOutputWeights[HiddenNodes][i] ;
        OutputWeights[HiddenNodes][i] += ChangeOutputWeights[HiddenNodes][i] ;
        for( j = 0 ; j < HiddenNodes ; j++ ) {
          ChangeOutputWeights[j][i] = LearningRate * Hidden[j] * OutputDelta[i] + Momentum * ChangeOutputWeights[j][i] ;
          OutputWeights[j][i] += ChangeOutputWeights[j][i] ;
        }
      }
    }

    // Periodically report to terminal
    ReportEvery1000 = ReportEvery1000 - 1;
    if (ReportEvery1000 == 0)
    {
      Serial.println(); 
      Serial.print ("TrainingCycle: ");
      Serial.print (TrainingCycle);
      Serial.print ("  Error = ");
      Serial.println (Error, 5);

      toTerminal();

      if (TrainingCycle==1) {
        ReportEvery1000 = 999;
      } else {
        ReportEvery1000 = 1000;
      }
    }    

    // Stop if the error is lower than success threshold
    if( Error < Success ) break ;  
  }
  
  Serial.println ();
  Serial.println(); 
  Serial.print ("TrainingCycle: ");
  Serial.print (TrainingCycle);
  Serial.print ("  Error = ");
  Serial.println (Error, 5);

  toTerminal();

  Serial.println ();  
  Serial.println ();
  Serial.println ("Training Set Solved! ");
  Serial.println ("--------"); 
  Serial.println ();
  Serial.println ();  
  while(true); // Halt execution once solved
}

void toTerminal() {
  for( p = 0 ; p < PatternCount ; p++ ) { 
    Serial.println(); 
    Serial.print ("Training Pattern: ");
    Serial.println (p);      
    Serial.print ("Input ");
    for( i = 0 ; i < InputNodes ; i++ ) {
      Serial.print (Input[p][i], DEC);
      Serial.print (" ");
    }
    Serial.print ("   Target ");
    for( i = 0 ; i < OutputNodes ; i++ ) {
      Serial.print (Target[p][i], DEC);
      Serial.print (" ");
    }

    // Compute hidden layer activations
    for( i = 0 ; i < HiddenNodes ; i++ ) {    
      Accum = HiddenWeights[InputNodes][i] ;
      for( j = 0 ; j < InputNodes ; j++ ) {
        Accum += Input[p][j] * HiddenWeights[j][i] ;
      }
      Hidden[i] = 1.0/(1.0 + exp(-Accum)) ;
    }

    // Compute output layer activations
    for( i = 0 ; i < OutputNodes ; i++ ) {    
      Accum = OutputWeights[HiddenNodes][i] ;
      for( j = 0 ; j < HiddenNodes ; j++ ) {
        Accum += Hidden[j] * OutputWeights[j][i] ;
      }
      Output[i] = 1.0/(1.0 + exp(-Accum)) ; 
    }
    
    Serial.print ("   Output ");
    for( i = 0 ; i < OutputNodes ; i++ ) {       
      Serial.print (Output[i], 5);
      Serial.print (" ");
    }
  }
}
