module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD CDN ||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE CDN ||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  name      : '|USA CDN ||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'world',
  script    : '93018.ddns.net/cdn.js',
  name      : '|WORLD 93018||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'europe',
  script    : '93018.ddns.net/cdn_europe.js',
  name      : '|EUROPE 93018||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'united states',
  script    : '93018.ddns.net/cdn_united_states.js',
  name      : '|USA 93018||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'world',
  script    : 'www.universalbit.it/cdn.js',
  name      : '|WORLD WWW UNBT||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'europe',
  script    : 'www.universalbit.it/cdn_europe.js',
  name      : '|EUROPE WWW UNBT||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'united states',
  script    : 'www.universalbit.it/cdn_united_states.js',
  name      : '|USA WWW UNBT||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'world',
  script    : 'universalbit.it/cdn.js',
  name      : '|WORLD UNIVERSALBIT||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'europe',
  script    : 'universalbit.it/cdn_europe.js',
  name      : '|EUROPE UNIVERSALBIT||',
  exec_mode : "cluster",
  instances : "max"
  },
  {
  name: 'united states',
  script    : 'universalbit.it/cdn_united_states.js',
  name      : '|USA UNIVERSALBIT||',
  exec_mode : "cluster",
  instances : "max"
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
