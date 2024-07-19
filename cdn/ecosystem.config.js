module.exports = {
  apps : [
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  args      : '',
  name      : '|EUROPE|-cdn.jsdelivr.net-|',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  args      : '',
  name      : '|USA|-cdn.jsdelivr.net-|',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  },
    
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  args      : '',
  name      : '|Italy|-cdn.jsdelivr.net-|',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  },

  {
  name: 'italy palermo',
  script    : 'cdn_italy_palermo.js',
  args      : '',
  name      : '|Italy Palermo|-cdn.jsdelivr.net-|',
  instances : "1",
  restart_delay: 3000,
  exec_mode : "cluster"
  }
]

}
