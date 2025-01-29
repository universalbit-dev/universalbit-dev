/* Run Bash Script */
var exec = require('node:child_process').exec
exec('echo $(pm2 restart all --restart-delay=30000)',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
    });

/* COPILOT EXPLAIN
This JavaScript file, `restart.js`, runs a command using Node.js's `child_process` module. Here's a breakdown of what it does:

1. **Require Child Process Module**: The `child_process` module is required to execute shell commands.
2. **Execute Bash Command**:
   - The script executes the command `pm2 restart all --restart-delay=30000` which restarts all applications managed by PM2 with a delay of 30 seconds between each restart.
   - It logs the standard output (`stdout`) and standard error (`stderr`) of the command.
3. **Error Handling**: If there is an error, it logs the error message.

This script is used to restart all PM2-managed applications with a specified delay between restarts.
*/
