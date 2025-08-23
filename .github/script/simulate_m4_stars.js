/**
 * File: simulate_m4_stars.js
 * Author: Universalbit Dev
 * License: MIT
 * Description:
 * Fetches the astronomical coordinates of M4 (NGC 6121) using the Noctua Sky API,
 * simulates stars within the cluster, and assigns each a unique deterministic ID.
 * Suitable for research, space economy, or crypto contexts.
 */

const axios = require('axios');
const crypto = require('crypto');

// Parameters
const m4Name = 'NGC 6121';
const apiUrl = `https://api.noctuasky.com/api/v1/skysources/name/${encodeURIComponent(m4Name)}`;
const numStars = 10000; // Demo: 10,000 stars. Increase as needed.

function simulateStar(clusterCenter, index) {
  // Spread stars randomly within 0.1 degrees of center (simple model)
  const ra = clusterCenter.ra + (Math.random() - 0.5) * 0.2;
  const dec = clusterCenter.dec + (Math.random() - 0.5) * 0.2;
  const mag = 10 + Math.random() * 6; // Magnitude between 10 and 16

  // Unique, deterministic ID based on cluster name, index, and star properties
  const rawId = `${m4Name}_${index}_${ra.toFixed(6)}_${dec.toFixed(6)}_${mag.toFixed(2)}`;
  const uniqueId = crypto.createHash('sha256').update(rawId).digest('hex').slice(0, 16);
  return { ra, dec, mag, id: `M4_${uniqueId}` };
}

axios.get(apiUrl)
  .then(response => {
    // Debug: Inspect API response structure
    console.log("Raw API response:", response.data);

    // Get RA/Dec from model_data
    const model = response.data.model_data;
    const ra = model.ra;
    const dec = model.de;

    if (ra === undefined || dec === undefined) {
      console.error("ERROR: Could not find RA/Dec in API response!");
      process.exit(1);
    }

    console.log(`M4 (NGC 6121) Center: RA=${ra}, Dec=${dec}`);

    // Simulate stars and assign unique IDs
    const stars = [];
    for (let i = 0; i < numStars; i++) {
      stars.push(simulateStar({ ra, dec }, i));
    }

    // Print sample stars
    console.log("Sample simulated star IDs:");
    stars.slice(0, 5).forEach(star =>
      console.log(`ID: ${star.id} | RA: ${star.ra.toFixed(6)} | Dec: ${star.dec.toFixed(6)} | Mag: ${star.mag.toFixed(2)}`)
    );

    // Optionally, export or process the full star list here
    // e.g., save to JSON file, use in blockchain registry, etc.
  })
  .catch(err => {
    console.error("API request failed:", err.message);
    process.exit(1);
  });
