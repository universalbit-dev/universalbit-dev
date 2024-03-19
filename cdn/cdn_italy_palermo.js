/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(globalping ping cdn.jsdelivr.net from italy+palermo --limit 10)',
    function (error, stdout, stderr) {
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });

