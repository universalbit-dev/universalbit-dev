const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');

function installDigiByte() {
  console.log("Updating package list...");
  execSync('sudo apt-get update', { stdio: 'inherit' });
  console.log("Installing dependencies...");
  execSync('sudo apt-get install -y wget unzip', { stdio: 'inherit' });

  // Updated URL and version for DigiByte Core
  const DGB_VERSION = 'v8.22.2';
  const DGB_FILENAME = 'digibyte-8.22.2-x86_64-linux-gnu.tar.gz';
  const DGB_URL = `https://github.com/DigiByte-Core/digibyte/releases/download/${DGB_VERSION}/${DGB_FILENAME}`;

  console.log("Downloading DigiByte Core...");
  execSync(`wget ${DGB_URL} -O /tmp/dgb.tar.gz`, { stdio: 'inherit' });

  console.log("Extracting DigiByte Core...");
  execSync('tar -xzf /tmp/dgb.tar.gz -C /tmp', { stdio: 'inherit' });

  // You may need to adjust the folder name if the extracted folder name changes
  const extractedDir = '/tmp/digibyte-8.22.2';

  console.log("Installing DigiByte Core binaries...");
  execSync(`sudo cp ${extractedDir}/bin/digibyted /usr/local/bin/`, { stdio: 'inherit' });
  execSync(`sudo cp ${extractedDir}/bin/digibyte-cli /usr/local/bin/`, { stdio: 'inherit' });

  console.log("DigiByte Core installation complete.");
}

try {
  installDigiByte();
} catch (error) {
  console.error("Installation failed:", error);
  process.exit(1);
}
