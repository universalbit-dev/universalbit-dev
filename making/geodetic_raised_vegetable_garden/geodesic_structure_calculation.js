/**
 * Geodesic Structure Calculation Script
 * 
 * This script calculates parameters for building a geodesic structure, 
 * such as the base area, number of triangles, wood pieces required, and triangle angles.
 * It supports customizable frequencies (e.g., 3V, 4V) and allows dynamic input for the dome radius and frequency.
 * 
 * Author: universalbit-dev
 * Repository: https://github.com/universalbit-dev/universalbit-dev
 * License: MIT
 */

const readline = require('readline');

// Constants
const PI = Math.PI;

// Function to calculate the surface area of the geodesic base
function calculateBaseArea(radius) {
    return PI * radius * radius; // A = π * r^2
}

// Function to calculate the number of triangles based on frequency
function calculateNumberOfTriangles(frequency) {
    // Approximation formula: 20 × frequency²
    return 20 * Math.pow(frequency, 2);
}

// Function to calculate the wood pieces needed
function calculateWoodPieces(numTriangles) {
    return numTriangles * 3; // Each triangle has 3 edges
}

// Function to calculate angles of equilateral triangles
function calculateTriangleAngles() {
    return 60; // Each angle in an equilateral triangle is 60 degrees
}

// Main function to calculate geodesic structure parameters
function calculateGeodesicStructure(radius, frequency) {
    if (frequency < 1) {
        throw new Error("Frequency must be a positive integer greater than or equal to 1.");
    }

    const baseArea = calculateBaseArea(radius);
    const numTriangles = calculateNumberOfTriangles(frequency);
    const woodPieces = calculateWoodPieces(numTriangles);
    const triangleAngles = calculateTriangleAngles();

    return {
        radius: radius.toFixed(2),
        frequency: frequency,
        baseArea: baseArea.toFixed(2), // Floor area in square units
        numTriangles: numTriangles,
        woodPieces: woodPieces,
        triangleAngles: triangleAngles
    };
}

// Set up readline interface for user input
const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout
});

// Prompt user for radius and validate input
function promptForRadius() {
    return new Promise((resolve) => {
        rl.question("Enter the radius of the geodesic structure in meters (e.g., 1, 0.5): ", (input) => {
            const radius = parseFloat(input);
            if (isNaN(radius) || radius <= 0) {
                console.log("❌ Invalid radius! Please enter a positive number greater than 0.");
                resolve(promptForRadius()); // Recursively prompt again for valid input
            } else {
                resolve(radius);
            }
        });
    });
}

// Prompt user for frequency and validate input
function promptForFrequency() {
    return new Promise((resolve) => {
        rl.question("Enter the frequency of the geodesic structure (e.g., 3 for 3V, 4 for 4V): ", (input) => {
            const frequency = parseInt(input, 10);
            if (isNaN(frequency) || frequency < 1) {
                console.log("❌ Invalid frequency! Please enter a positive integer greater than or equal to 1.");
                resolve(promptForFrequency()); // Recursively prompt again for valid input
            } else {
                resolve(frequency);
            }
        });
    });
}

// Main program flow
async function main() {
    try {
        console.log("🌍 Welcome to the Geodesic Structure Calculator!");
        console.log("------------------------------------------------\n");

        // Get user inputs for radius and frequency
        const radius = await promptForRadius();
        const frequency = await promptForFrequency();

        // Calculate results
        const result = calculateGeodesicStructure(radius, frequency);

        // Display results
        console.log("\n✅ Geodesic Structure Calculation Results:");
        console.log("------------------------------------------------");
        console.log(`🌟 Radius: ${result.radius} meters`);
        console.log(`🌟 Frequency: ${result.frequency}V`);
        console.log(`🌟 Base Area: ${result.baseArea} m²`);
        console.log(`🌟 Number of Triangles: ${result.numTriangles}`);
        console.log(`🌟 Number of Wood Pieces: ${result.woodPieces}`);
        console.log(`🌟 Triangle Angles: ${result.triangleAngles}°\n`);
    } catch (error) {
        console.error("❌ An error occurred:", error.message);
    } finally {
        rl.close(); // Ensure the readline interface is closed
        console.log("👋 Thank you for using the Geodesic Structure Calculator!");
    }
}

main(); // Start the program
