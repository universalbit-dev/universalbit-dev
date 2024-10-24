const { addon: ov } = require('openvino-node');
const { Chess } = require('chess.js')

thegameofchess=[];

function fxchess() {
  const chess = new Chess()
  while (!chess.isGameOver()) {
  const moves = chess.moves()
  const move = moves[Math.floor(Math.random() * moves.length)]
  chess.move(move)
}
return console.log(chess.pgn())
}

console.log("Random game of Chess");fxchess();

module.exports = thegameofchess;




