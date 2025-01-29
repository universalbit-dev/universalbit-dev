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

/* COPILOT EXPLAIN
This JavaScript file, autoclean.js, runs a command using Node.js's child_process module. Here's a breakdown of what it does:

    Require Child Process Module: The child_process module is required to execute shell commands.
    Execute Bash Command:
        The script executes the command rm ~/.pm2/logs/*.log which removes all log files in the ~/.pm2/logs directory.
        It logs the standard output (stdout) and standard error (stderr) of the command.
    Error Handling: If there is an error, it logs the error message.

This script is used to clean up log files in the PM2 logs directory.
*/
