require("noop");
function noop() {}
function throwop(err) {
  if (err) {
    throw err;
  }
}
