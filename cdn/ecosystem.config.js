module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|Globalping CDN ||',
  exec_mode : "cluster",
  instances : "1"
  },
  {
  name: 'unbt',
  script    : 'universalbit.js',
  name      : '|UNBT||',
  exec_mode : "fork"
  },
  {
  name: 'restart',
  script    : 'restart.js',
  name      : '|Restart||',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  },
  {
  name: 'autoclean',
  script    : 'autoclean.js',
  name      : '|AutoClean||',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  }
]
}
