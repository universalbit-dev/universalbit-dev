/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(pm2 restart all)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });