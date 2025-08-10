/**
 * This file is licensed under the Creative Commons Attribution-NonCommercial 3.0 IGO (CC BY-NC 3.0 IGO) license.
 * More details at: https://www.cosmos.esa.int/web/gaia-users/license
 */

/**
 * File: gaia_ngc6121_reference.js
 * Author: Universalbit Dev
 * Description:
 * This script generates unique Gaia NGC 6121 reference numbers by combining a
 * user-defined prefix with a randomly generated number of specified digits.
 * The reference number is intended for unique identification within the Gaia dataset.
 * The process repeats continuously for demonstration or batch processing purposes.
 */

function generateGaiaReference(prefix, numDigits) {
    const max = Math.pow(10, numDigits);
    const randomNumber = Math.floor(Math.random() * max)
        .toString()
        .padStart(numDigits, '0');
    return `${prefix}${randomNumber}`;
}

// Parameters for NGC 6121
const prefix = "6121";
const numDigits = 14;

// Continuously generate and print new references every 5 seconds
function runContinuously() {
    setInterval(() => {
        const gaiaReference = generateGaiaReference(prefix, numDigits);
        console.log("Generated Gaia NGC 6121 Reference:", gaiaReference);
    }, 5000);
}

runContinuously();
