require("noop");

function noop() {}

function throwop(err) {
  if (err) {
    throw err;
  }
}

function doop(callback, args, context) {
  if ("function" === typeof callback) {
    callback.apply(context, args);
  }
}
