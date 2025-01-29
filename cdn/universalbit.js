/* GlobalPing UniversalBit Project */
var universalbit = {};
var unbtit = require('node:child_process').exec;

unbtit('echo $(ping cdn.jsdelivr.net -i 10)', function (error, stdout, stderr) {
    console.log('stdout: ' + stdout);
    console.log('stderr: ' + stderr);
    if (error !== null) {
        console.log('exec error: ' + error);
    }
});

module.exports = universalbit;
