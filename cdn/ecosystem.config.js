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
  name: 'france',
  script    : 'cdn_france.js',
  name      : '|France||',
  instances : "1",
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'cdn_germany.js',
  name      : '|Germany||',
  instances : "1",
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'cdn_spain.js',
  name      : '|Spain||',
  instances : "1",
  exec_mode : "cluster"
  }


]

}
