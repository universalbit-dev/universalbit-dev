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
