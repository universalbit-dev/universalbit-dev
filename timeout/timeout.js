const humanizeDuration = require("humanize-duration"); //easy milliseconds,seconds,days,months,years

console.log('Function sleep when Buy or Sell -- Activated --');

time = 12000

function zzzsleep() {
    setTimeout(() => {
        console.log('Buy or Sell submitted...zzz');
    }, time); 
}
zzzsleep();
console.log('-- Function zzzsleep --');

/*

v2 with nodejs promises
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
