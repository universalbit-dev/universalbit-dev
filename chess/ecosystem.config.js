module.exports = {
  apps : [
  {
  name: 'chess',
  script    : 'the_random_game_of_chess.js',
  name      : '|CHESS||',
  exec_mode: 'fork',
  exp_backoff_restart_delay: 10000
  }
]

}
