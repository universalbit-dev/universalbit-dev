module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD||',
  instances : "1",
  exp_backoff_restart_delay: 5000,
  exec_mode : ""
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE||',
  exp_backoff_restart_delay: 5,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  name      : '|USA||',
  exp_backoff_restart_delay: 5,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  name      : '|Italy||',
  exp_backoff_restart_delay: 50,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'france',
  script    : 'cdn_france.js',
  name      : '|France||',
  exp_backoff_restart_delay: 5000,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'germany',
  script    : 'cdn_germany.js',
  name      : '|Germany||',
  exp_backoff_restart_delay: 5000,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'spain',
  script    : 'cdn_spain.js',
  name      : '|Spain||',
  exp_backoff_restart_delay: 5000,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'restart',
  script    : 'restart.js',
  name      : '|Restart||',
  exp_backoff_restart_delay: 50000,
  exec_mode : ""
  }
]

}
