var _ = require('lodash/core');
require("noop");
function noop() {}
function throwop(err) {
  if (err) {
    throw err;
  }
}
