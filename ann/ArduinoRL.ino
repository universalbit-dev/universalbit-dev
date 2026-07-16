/*
 * ArduinoRL.ino
 * A simple Q-learning (Reinforcement Learning) agent on Arduino.
 * 
 * Environment: 1D Grid World with 6 states (0 to 5)
 * [ State 0 ] <-> [ State 1 ] <-> [ State 2 ] <-> [ State 3 ] <-> [ State 4 ] <-> [ State 5 (GOAL) ]
 * Actions: 0 (Go Left), 1 (Go Right)
 */

#include <Arduino.h>

#define NUM_STATES 6
#define NUM_ACTIONS 2

// RL Hyperparameters
const float alpha = 0.1;          // Learning rate
const float discount_gamma = 0.9; // Discount factor (renamed to prevent math.h collision)
const float epsilon = 0.2;        // Exploration rate (epsilon-greedy)

// Q-Table: States x Actions
float Q[NUM_STATES][NUM_ACTIONS];

// Current state of the agent
int currentState = 0;
const int goalState = 5;
unsigned long stepCount = 0;
unsigned int episodeCount = 1;

void setup() {
    Serial.begin(115200);
    while (!Serial); // Wait for serial port to connect
    
    // Seed random number generator
    randomSeed(analogRead(0));

    // Initialize Q-table with zeros
    for (int s = 0; s < NUM_STATES; s++) {
        for (int a = 0; a < NUM_ACTIONS; a++) {
            Q[s][a] = 0.0;
        }
    }

    Serial.println("=================================================");
    Serial.println("      Arduino Q-Learning Agent Initialized       ");
    Serial.println("=================================================");
    Serial.print("Episode: ");
    Serial.println(episodeCount);
}

void loop() {
    int action;

    // 1. Action Selection (Epsilon-Greedy Strategy)
    float randVal = (float)random(0, 100) / 100.0;
    if (randVal < epsilon) {
        // Explore: select random action
        action = random(0, NUM_ACTIONS);
    } else {
        // Exploit: select action with best Q-value
        if (Q[currentState][0] >= Q[currentState][1]) {
            action = 0;
        } else {
            action = 1;
        }
    }

    // 2. Environment Physics: Take Action & Observe Next State
    int nextState = currentState;
    if (action == 0) { // Move Left
        if (currentState > 0) nextState = currentState - 1;
    } else {           // Move Right
        if (currentState < NUM_STATES - 1) nextState = currentState + 1;
    }

    // 3. Define Rewards
    float reward = -1.0; // Default step penalty (encourages speed)
    if (nextState == goalState) {
        reward = 100.0;  // High reward for reaching the goal state
    }

    // 4. Update Q-Value using the Bellman Equation:
    // Q(s, a) = Q(s, a) + alpha * (reward + discount_gamma * max(Q(s', a')) - Q(s, a))
    float maxNextQ = (Q[nextState][0] > Q[nextState][1]) ? Q[nextState][0] : Q[nextState][1];
    Q[currentState][action] = Q[currentState][action] + alpha * (reward + (discount_gamma * maxNextQ) - Q[currentState][action]);

    // Update state tracking
    currentState = nextState;
    stepCount++;

    // 5. Goal State Handshake
    if (currentState == goalState) {
        Serial.print("Goal Reached! Steps taken: ");
        Serial.print(stepCount);
        Serial.println(" | Current Q-Table state mappings:");
        
        // Print updated Q-Table
        for (int s = 0; s < NUM_STATES; s++) {
            Serial.print("State ");
            Serial.print(s);
            Serial.print(" -> [Left: ");
            Serial.print(Q[s][0], 2);
            Serial.print(" | Right: ");
            Serial.print(Q[s][1], 2);
            Serial.println("]");
        }
        Serial.println("-------------------------------------------------");

        // Reset for the next training cycle
        currentState = 0;
        stepCount = 0;
        episodeCount++;
        
        Serial.print("Episode: ");
        Serial.println(episodeCount);
        delay(1000); // Wait briefly before starting the next cycle
    }
}
