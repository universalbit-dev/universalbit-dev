module.exports = {
  apps : [
  {
    name: 'unbt_globalping_cdn',
    script    : 'universalbit.js',
    name      : '| UNBT ||',
    exec_mode : "fork",
    restart_delay: 30000,
    env: {NODE_ENV: "development",},
    env_production: {NODE_ENV: "production",}
  }
]
}

