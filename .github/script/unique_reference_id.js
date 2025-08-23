/**
 * This file is licensed under the GNU General Public License v3.0 (GPL-3.0).
 * You may redistribute, modify, and/or use this file under the terms of the GPL-3.0 license.
 * For more details, see: https://www.gnu.org/licenses/gpl-3.0.html
 */

/**
 * File: unique_reference_id.js
 * Author: Universalbit-dev
 * Description:
 * This script simulates the generation of unique Gaia references for major celestial objects.
 * 
 * Features:
 * - Randomly selects a major known celestial object from a preset array, including clusters and nebulae.
 * - Fetches accurate astronomical coordinates (RA, Dec) for the selected object from the Noctua Sky API using axios.
 * - Generates a unique reference ID by combining a preset prefix and a random numeric sequence.
 * - Outputs the selected object's name, coordinates, and generated reference in a standardized format,
 *   suitable for automated pipelines, testing, or integration into astronomical simulations.
 *
 * Usage:
 * - Run as a standalone Node.js script to generate simulated Gaia references with real-time API data.
 */

const axios = require('axios');

// Preset array of major celestial objects for API calls
const celestialObjects = [
    { name: "NGC 6121 (M4)", prefix: "6121", apiName: "M 4" },
    { name: "NGC 104 (47 Tuc)", prefix: "0104", apiName: "47 Tuc" },
    { name: "NGC 5139 (Omega Centauri)", prefix: "5139", apiName: "Omega Centauri" },
    { name: "NGC 7078 (M15)", prefix: "7078", apiName: "M 15" },
    { name: "NGC 5272 (M3)", prefix: "5272", apiName: "M 3" },
    { name: "NGC 6205 (M13)", prefix: "6205", apiName: "M 13" },
    { name: "NGC 5904 (M5)", prefix: "5904", apiName: "M 5" },
    { name: "NGC 6341 (M92)", prefix: "6341", apiName: "M 92" },
    { name: "NGC 288", prefix: "0288", apiName: "NGC 288" },
    { name: "NGC 1851", prefix: "1851", apiName: "NGC 1851" },
    { name: "NGC 2419", prefix: "2419", apiName: "NGC 2419" },
    { name: "NGC 6397", prefix: "6397", apiName: "NGC 6397" },
    { name: "NGC 6752", prefix: "6752", apiName: "NGC 6752" },
    { name: "NGC 7099 (M30)", prefix: "7099", apiName: "M 30" },
    { name: "NGC 7089 (M2)", prefix: "7089", apiName: "M 2" },
    { name: "NGC 5466", prefix: "5466", apiName: "NGC 5466" },
    { name: "NGC 5024 (M53)", prefix: "5024", apiName: "M 53" },
    { name: "NGC 5053", prefix: "5053", apiName: "NGC 5053" },
    { name: "NGC 6535", prefix: "6535", apiName: "NGC 6535" },
    { name: "NGC 6934", prefix: "6934", apiName: "NGC 6934" },
    { name: "NGC 7006", prefix: "7006", apiName: "NGC 7006" },
    { name: "NGC 7492", prefix: "7492", apiName: "NGC 7492" },
    { name: "NGC 6229", prefix: "6229", apiName: "NGC 6229" },
    // Famous open clusters and nebulae
    { name: "Pleiades (M45)", prefix: "M45", apiName: "M 45" },
    { name: "Orion Nebula (M42)", prefix: "M42", apiName: "M 42" },
    { name: "Hyades", prefix: "Hyad", apiName: "Hyades" },
    { name: "Lagoon Nebula (M8)", prefix: "M8", apiName: "M 8" },
    { name: "NGC 884 (Double Cluster)", prefix: "0884", apiName: "NGC 884" },
    { name: "NGC 869 (Double Cluster)", prefix: "0869", apiName: "NGC 869" },
    { name: "NGC 3532 (Wishing Well Cluster)", prefix: "3532", apiName: "NGC 3532" },
    { name: "NGC 4755 (Jewel Box)", prefix: "4755", apiName: "NGC 4755" },
];

// Pick a random celestial object
const selected = celestialObjects[Math.floor(Math.random() * celestialObjects.length)];

// Generate a unique reference for the selected object
function generateReference(prefix, numDigits = 14) {
    const max = Math.pow(10, numDigits);
    const randomNumber = Math.floor(Math.random() * max)
        .toString()
        .padStart(numDigits, '0');
    return `${prefix}${randomNumber}`;
}

// Fetch coordinates from Noctua Sky API using axios
async function fetchCoordinates(apiName) {
    const encodedName = encodeURIComponent(apiName);
    const url = `https://api.noctuasky.com/api/v1/skysources/name/${encodedName}`;
    try {
        const response = await axios.get(url, {
            timeout: 5000,
            headers: { 'Accept': 'application/json' }
        });
        const data = response.data;
        if (data && data.model_data && typeof data.model_data.ra !== "undefined" && typeof data.model_data.de !== "undefined") {
            return { ra: data.model_data.ra, dec: data.model_data.de };
        } else {
            console.warn(`[WARNING] API response missing model_data for ${apiName}, using fallback values.`);
            return { ra: 0, dec: 0 };
        }
    } catch (error) {
        console.error(`[ERROR] Failed to fetch coordinates for ${apiName}:`, error.message);
        return { ra: 0, dec: 0 };
    }
}

(async () => {
    const uniqueReference = generateReference(selected.prefix);
    const coords = await fetchCoordinates(selected.apiName);

    console.log(`[DATA]: Selected Celestial Object: ${selected.name}`);
    console.log(`[DATA]: Astronomical Coordinates RA: ${coords.ra} DEC: ${coords.dec}`);
    console.log(`[DATA]: Generated Unique Reference: ${uniqueReference}`);

    process.exit(0);
})();
