module.exports = {
  apps : [
  {
  name: 'world',
  script    : 'cdn.js',
  name      : '|WORLD CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'cdn_europe.js',
  name      : '|EUROPE CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'cdn_united_states.js',
  name      : '|USA CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : 'cdn_italy.js',
  name      : '|Italy CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : 'cdn_france.js',
  name      : '|France CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'cdn_germany.js',
  name      : '|Germany CDN||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'cdn_spain.js',
  name      : '|Spain CDN ||',
  exec_mode : "cluster"
  },
  {
  name: 'world',
  script    : 'www.universalbit.it/cdn.js',
  name      : '|WORLD WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'www.universalbit.it/cdn_europe.js',
  name      : '|EUROPE WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'www.universalbit.it/cdn_united_states.js',
  name      : '|USA WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : 'www.universalbit.it/cdn_italy.js',
  name      : '|Italy WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : 'www.universalbit.it/cdn_france.js',
  name      : '|France WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'www.universalbit.it/cdn_germany.js',
  name      : '|Germany WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'www.universalbit.it/cdn_spain.js',
  name      : '|Spain WWW UNBT||',
  exec_mode : "cluster"
  },
  {
  name: 'world',
  script    : 'universalbit.it/cdn.js',
  name      : '|WORLD UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : 'universalbit.it/cdn_europe.js',
  name      : '|EUROPE UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : 'universalbit.it/cdn_united_states.js',
  name      : '|USA UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : 'universalbit.it/cdn_italy.js',
  name      : '|Italy UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : 'universalbit.it/cdn_france.js',
  name      : '|France UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : 'universalbit.it/cdn_germany.js',
  name      : '|Germany UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : 'universalbit.it/cdn_spain.js',
  name      : '|Spain UNIVERSALBIT||',
  exec_mode : "cluster"
  },
  {
  name: 'world',
  script    : '93018.ddns.net/cdn.js',
  name      : '|WORLD 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'europe',
  script    : '93018.ddns.net/cdn_europe.js',
  name      : '|EUROPE 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'united states',
  script    : '93018.ddns.net/cdn_united_states.js',
  name      : '|USA 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'italy',
  script    : '93018.ddns.net/cdn_italy.js',
  name      : '|Italy 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'france',
  script    : '93018.ddns.net/cdn_france.js',
  name      : '|France 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'germany',
  script    : '93018.ddns.net/cdn_germany.js',
  name      : '|Germany 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'spain',
  script    : '93018.ddns.net/cdn_spain.js',
  name      : '|Spain 93018||',
  exec_mode : "cluster"
  },
  {
  name: 'restart',
  script    : 'restart.js',
  name      : '|Restart||',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  }
]

}
