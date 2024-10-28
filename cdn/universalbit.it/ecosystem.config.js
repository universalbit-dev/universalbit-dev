module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  name      : '|USA UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  name      : '|Italy UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : 'cdn_france.js',
  name      : '|France UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'cdn_germany.js',
  name      : '|Germany UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'cdn_spain.js',
  name      : '|Spain UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'restart',
  script    : 'restart.js',
  name      : '|Restart||',
  exp_backoff_restart_delay: 5000000,
  exec_mode : "fork"
  }
]

}
