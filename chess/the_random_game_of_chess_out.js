const { addon: ov } = require('openvino-node');
const { Chess } = require('chess.js');
var fs = require('fs/promises');
thegameofchess=[];

async function fxchess() {
  const chess = new Chess()
  while (!chess.isGameOver()) {
  const moves = chess.moves()
  const move = moves[Math.floor(Math.random() * moves.length)]
  chess.move(move)
}  
return await fs.writeFile("chessout.json",JSON.stringify(chess.pgn()));
}
fxchess();
module.exports = thegameofchess;




