module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD||',
  instances : "1",
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE||',
  instances : "1",
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  args      : '',
  name      : '|USA||',
  exec_mode : "cluster"
  },
    
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  args      : '',
  name      : '|Italy||',
  exec_mode : "cluster"
  },

  {
  name: 'italy palermo',
  script    : 'cdn_italy_palermo.js',
  name      : '|Italy Palermo||',
  instances : "1",
  exec_mode : "cluster"
  }
]

}
