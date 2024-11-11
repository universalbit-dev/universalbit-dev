/* GlobalPing UniversalBit Project */
var universalbit = {};
var unbtit = require('node:child_process').exec
unbtit('echo $(globalping dns universalbitcdn.it from world --limit 5  )',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
});

var unbtrepo = require('node:child_process').exec
unbtrepo('echo $(globalping dns universalbit.it from world --limit 5  )',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
});

var unbtsite = require('node:child_process').exec
unbtsite('echo $(globalping dns www.universalbit.it from world --limit 5  )',
    function (error, stdout, stderr) {
        console.log('stdout:'+ stdout);
        console.log('stderr: ' + stderr);
        if (error !== null) {
             console.log('exec error: ' + error);
        }
});

module.exports = universalbit;
