/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(globalping dns universalbit.it --limit 3)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
