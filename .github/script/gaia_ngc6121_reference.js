/**
 * This file is licensed under the Creative Commons Attribution-NonCommercial 3.0 IGO (CC BY-NC 3.0 IGO) license.
 * More details at: https://www.cosmos.esa.int/web/gaia-users/license
 */

/**
 * File: gaia_ngc6121_reference.js
 * Author: Universalbit Dev
 * Description:
 * This script generates a unique Gaia NGC 6121 reference number by combining a
 * user-defined prefix with a randomly generated number of specified digits.
 * The reference number is intended for unique identification within the Gaia dataset.
 */

function generateGaiaReference(prefix, numDigits) {
    // Generate a random number with the specified number of digits
    const randomNumber = Math.floor(
        Math.random() * Math.pow(10, numDigits)
    ).toString().padStart(numDigits, '0');

    // Combine prefix and random number
    return `${prefix}${randomNumber}`;
}

// Parameters - update as needed for NGC 6121
const prefix = "6121";
const numDigits = 14;

// Generate the Gaia NGC 6121 reference number
const gaiaReference = generateGaiaReference(prefix, numDigits);
console.log("Generated Gaia NGC 6121 Reference:", gaiaReference);
