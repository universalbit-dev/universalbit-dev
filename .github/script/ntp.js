const { execSync } = require('child_process');
const fs = require('fs');

function installNTP() {
  console.log('Installing NTP...');
  execSync('sudo apt-get update && sudo apt-get install -y ntp', { stdio: 'inherit' });
}

function configureNTP() {
  const config = `
server 0.pool.ntp.org iburst
server 1.pool.ntp.org iburst
server 2.pool.ntp.org iburst
server 3.pool.ntp.org iburst

driftfile /var/lib/ntp/ntp.drift

restrict default kod nomodify notrap nopeer noquery
restrict 127.0.0.1
restrict ::1
  `;
  console.log('Updating /etc/ntp.conf...');
  fs.writeFileSync('/tmp/ntp.conf', config); // Write to tmp first
  execSync('sudo mv /tmp/ntp.conf /etc/ntp.conf');
}

function reconfigureNTP() {
  console.log('Reconfiguring NTP...');
  execSync('sudo dpkg-reconfigure ntp', { stdio: 'inherit' });
}

installNTP();
configureNTP();
reconfigureNTP();

console.log('NTP setup completed.');
