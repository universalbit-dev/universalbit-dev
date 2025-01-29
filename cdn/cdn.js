//command line globalping (net optimizer):    https://globalping.io/cli

/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(globalping dns cdn.jsdelivr.net --limit 3)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });

/*
COPILOT EXPLAIN
This JavaScript file, cdn.js, runs a command using Node.js's child_process module. Here's a breakdown of what it does:

    Comment: It includes a comment providing a link to the Globalping CLI documentation.
    Require Child Process Module: The child_process module is required to execute shell commands.
    Execute Globalping Command:
        The script executes the command globalping dns cdn.jsdelivr.net --limit 3 which performs a DNS resolution query for cdn.jsdelivr.net using Globalping, limiting the results to 3.
        It logs the standard output (stdout) and standard error (stderr) of the command.
    Error Handling: If there is an error, it logs the error message.

This script is used to perform a DNS query and log the results.
*/
