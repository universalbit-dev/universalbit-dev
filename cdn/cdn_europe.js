/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(globalping dns cdn.jsdelivr.net from europe --limit 3)',
    function (error, stdout, stderr) {
        console.log('\x1b[33m%s\x1b[0m','Your Ip:' + stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });
