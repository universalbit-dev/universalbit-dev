module.exports = {
  apps : [
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  args      : '',
  name      : '|EUROPE||',
  instances : "1",
  restart_delay: 3000,
  autorestart: true,
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  args      : '',
  name      : '|USA||',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  },
    
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  args      : '',
  name      : '|Italy||',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  },

  {
  name: 'italy palermo',
  script    : 'cdn_italy_palermo.js',
  name      : '|Italy Palermo||',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  }
]

}
