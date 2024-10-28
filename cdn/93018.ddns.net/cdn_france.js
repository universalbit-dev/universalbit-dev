/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(globalping dns 93018.ddns.net from france --limit 10)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
