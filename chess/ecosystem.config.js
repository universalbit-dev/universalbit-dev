module.exports = {
  apps : [{
    name   : "chess",
    script : "./the_random_game_of_chess.js",
    instances : "max",
    exec_mode : "cluster"
  }]
}
