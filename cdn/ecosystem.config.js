module.exports = {
  apps : [
  {
  name      : '|Globalping CDN ||',
  script    : 'cdn.js',
  exec_mode : "cluster",
  instances : "1"
  },
  {
  name      : '|UNBT||',
  script    : 'universalbit.js',
  exec_mode : "fork"
  },
  {
  name      : '|Restart||',
  script    : 'restart.js',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  },
  {
  name      : '|AutoClean||',
  script    : 'autoclean.js',
  exp_backoff_restart_delay: 500000,
  exec_mode : "fork"
  }
]
}

/* COPILOT EXPLAIN
This ecosystem.config.js file is a configuration for PM2, a process manager for Node.js applications. It defines the settings for managing multiple applications. Here's what it does:

    Export Configuration: The file exports a configuration object with a key apps, which is an array of application configurations.
    Application Configurations:
        App 1: Globalping CDN:
            name: '|Globalping CDN ||'
            script: 'cdn.js' (the script to run)
            exec_mode: 'cluster' (runs in cluster mode)
            instances: '1' (number of instances to run)
        App 2: UNBT:
            name: '|UNBT||'
            script: 'universalbit.js' (the script to run)
            exec_mode: 'fork' (runs in fork mode)
        App 3: Restart:
            name: '|Restart||'
            script: 'restart.js' (the script to run)
            exp_backoff_restart_delay: 500000 (delay before restarting the app on failure)
            exec_mode: 'fork' (runs in fork mode)
        App 4: AutoClean:
            name: '|AutoClean||'
            script: 'autoclean.js' (the script to run)
            exp_backoff_restart_delay: 500000 (delay before restarting the app on failure)
            exec_mode: 'fork' (runs in fork mode)

This configuration sets up four applications to be managed by PM2 with specific settings for each application's execution mode, script, and restart behavior.

*/
