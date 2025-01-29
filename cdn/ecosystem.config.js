module.exports = {
  apps : [
  {
  name      : '|Globalping CDN ||',
  script    : 'cdn.js',
  exec_mode : "cluster",
  instances : "1"
  },
  {
  name      : '|UNBT||',
  script    : 'universalbit.js',
  exec_mode : "fork"
  },
  {
  name      : '|Restart||',
  script    : 'restart.js',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  },
  {
  name      : '|AutoClean||',
  script    : 'autoclean.js',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  }
]
}
