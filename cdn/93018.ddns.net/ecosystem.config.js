module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  name      : '|USA 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  name      : '|Italy 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : 'cdn_france.js',
  name      : '|France 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'cdn_germany.js',
  name      : '|Germany 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'cdn_spain.js',
  name      : '|Spain 93018||',
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
