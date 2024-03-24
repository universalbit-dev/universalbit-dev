const ProgressBar = require('console-progress-bar');
const { setTimeout: setTimeoutPromise } = require('node:timers/promises');

time =12000; //<== take your time (12 seconds)

console.log('Function zzzsleep -- Activated --');
const progressBar = new ProgressBar({ maxValue: 100 });

function progressbar()
{
 progressBar.addValue(1);
}
setInterval(progressbar, time/100);

function zzzsleep() {
    setTimeout(() => {
        console.log('  -- done --');process. exit();
    }, time);
}
zzzsleep();console.log('  -- wait --');

