const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Path to the chess project directory
const chessDir = path.join(__dirname, '../../chess');

function installDependencies() {
  if (fs.existsSync(path.join(chessDir, 'package.json'))) {
    console.log('Installing Node.js dependencies for chess project...');
    execSync('npm install', { cwd: chessDir, stdio: 'inherit' });
  } else if (fs.existsSync(path.join(chessDir, 'requirements.txt'))) {
    console.log('Installing Python dependencies for chess project...');
    execSync('pip install -r requirements.txt', { cwd: chessDir, stdio: 'inherit' });
  } else {
    console.log('No recognized dependency file found in chess project.');
  }
}

function runTests() {
  if (fs.existsSync(path.join(chessDir, 'package.json'))) {
    console.log('Running Node.js tests for chess project...');
    execSync('npm test', { cwd: chessDir, stdio: 'inherit' });
  } else if (fs.existsSync(path.join(chessDir, 'pytest.ini')) || fs.existsSync(path.join(chessDir, 'tests'))) {
    console.log('Running Python tests for chess project...');
    execSync('pytest', { cwd: chessDir, stdio: 'inherit' });
  } else {
    console.log('No test configuration found for chess project.');
  }
}

function main() {
  console.log('=== Chess Project Setup Script ===');
  installDependencies();
  runTests();
  console.log('=== Chess Project Setup Complete ===');
}

main();
