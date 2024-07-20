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
  name: 'italy_roma',
  script    : 'cdn_italy_roma.js',
  name      : '|Italy|Roma|',
  exp_backoff_restart_delay: 500,
  instances : "1",
  exec_mode : ""
  },

  {
  name: 'italy_palermo',
  script    : 'cdn_italy_palermo.js',
  name      : '|Italy|Palermo|',
  exp_backoff_restart_delay: 500,
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
  name: 'france_paris',
  script    : 'cdn_france_paris.js',
  name      : '|France|Paris|',
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
  name: 'germany_berlin',
  script    : 'cdn_germany_berlin.js',
  name      : '|Germany|Berlin|',
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
  name: 'spain_madrid',
  script    : 'cdn_spain_madrid.js',
  name      : '|Spain|Madrid|',
  exp_backoff_restart_delay: 500,
  instances : "1",
  exec_mode : ""
  },
  {
  name: 'restart',
  script    : 'restart.js',
  name      : '|Restart||',
  exp_backoff_restart_delay: 50000,
  instances : "1",
  exec_mode : ""
  }

]

}
