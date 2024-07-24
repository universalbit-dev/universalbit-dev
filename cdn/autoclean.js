/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(rm ~/.pm2/logs/*.log)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
