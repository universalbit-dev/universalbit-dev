const ProgressBar = require('console-progress-bar');

time =12000; //<== take your time (12 seconds)

console.log('Function zzzsleep -- Activated --');
const progressBar = new ProgressBar({ maxValue: 100 });


function zzzsleep() {
    setTimeout(() => {
        console.log('-- done --');
    }, time); 
}
zzzsleep();progressBar.addValue(1);
console.log('  -- wait --');
