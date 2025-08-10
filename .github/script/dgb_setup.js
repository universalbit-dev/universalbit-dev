const { execSync } = require('child_process');

function installDigiByte() {
  try {
    console.log('Updating package list...');
    execSync('sudo apt-get update', { stdio: 'inherit' });

    console.log('Installing dependencies...');
    execSync('sudo apt-get install -y wget unzip', { stdio: 'inherit' });

    // Download DigiByte Core (replace version as needed)
    console.log('Downloading DigiByte Core...');
    execSync(
      'wget wget https://github.com/DigiByte-Core/digibyte/releases/download/v8.22.2/digibyte-8.22.2-x86_64-linux-gnu.tar.gz',
      { stdio: 'inherit' }
    );

    console.log('Extracting DigiByte Core...');
    execSync('tar -xzf /tmp/dgb.tar.gz -C /tmp', { stdio: 'inherit' });

    // Move binaries to /usr/local/bin (customize path as needed)
    console.log('Installing DigiByte binaries...');
    execSync('sudo cp /tmp/digibyte-8.22.0/bin/* /usr/local/bin/', { stdio: 'inherit' });

    console.log('DigiByte Core installed successfully.');
  } catch (error) {
    console.error('Installation failed:', error);
    process.exit(1);
  }
}

function printNextSteps() {
  console.log(`
DigiByte Core is installed!

- To start DigiByte: digibyted
- To check version: digibyted --version

Refer to blockchain/digibyte/readme.md for detailed configuration options.
`);
}

installDigiByte();
printNextSteps();
