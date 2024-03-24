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

/*

v2 with nodejs promise
//https://www.linkedin.com/pulse/nodejs-16-settimeout-asyncawait-igor-gonchar
import {setTimeout} from "timers/promises";
console.log('Function sleep when Buy or Sell -- Activated --');
async function newStyleDelay() {
   await setTimeout(5000);
   console.log('It will be printed 3-rd with delay');
}
newStyleDelay();
console.log('It will be printed 2-nd');

*/
