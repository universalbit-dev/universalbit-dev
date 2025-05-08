/**
 * This file is licensed under the Creative Commons Attribution-NonCommercial 3.0 IGO (CC BY-NC 3.0 IGO) license.
 * 
 * You are free to:
 * - Share: Copy and redistribute the material in any medium or format.
 * - Adapt: Remix, transform, and build upon the material.
 * 
 * Under the following terms:
 * - Attribution: You must give appropriate credit, provide a link to the license, 
 *   and indicate if changes were made. You may do so in any reasonable manner, 
 *   but not in any way that suggests the licensor endorses you or your use.
 * - NonCommercial: You may not use the material for commercial purposes.
 * 
 * More details about this license are available at:
 * https://www.cosmos.esa.int/web/gaia-users/license
 */

/**
 * File: gaia_mission_dr2_reference.js
 * Author: Universalbit Dev
 * Description:
 * This script generates a unique Gaia Mission DR2 reference number by combining a 
 * user-defined prefix with a randomly generated number of specified digits. 
 * The reference number is intended for unique identification within the Gaia dataset.
 *
 * Usage:
 * 1. Set the `prefix` variable to the desired prefix string.
 * 2. Set the `numDigits` variable to the desired number of random digits.
 * 3. Run the script to generate and display the reference number in the console.
 *
 * Example:
 * Prefix: "60454"
 * Number of Digits: 14
 * Output: "6045412345678901234"
 *
 * Notes:
 * - Ensure the prefix and number of digits meet the requirements of your use case.
 * - Random numbers are generated using Math.random(), which may not be suitable 
 *   for cryptographic purposes.
 */


function generateGaiaReference(prefix, numDigits) {
    // Generate a random number with the specified number of digits
    const randomNumber = Math.floor(
        Math.random() * Math.pow(10, numDigits)
    ).toString().padStart(numDigits, '0');
    
    // Combine prefix and random number
    return `${prefix}${randomNumber}`;
}

// Parameters
const prefix = "60454"; // Your prefix
const numDigits = 14;   // Number of random digits

// Generate the Gaia mission DR2 reference number
const gaiaReference = generateGaiaReference(prefix, numDigits);
console.log("Generated Gaia mission DR2 Reference:", gaiaReference);
