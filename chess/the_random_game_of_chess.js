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
return await fs.appendFile("chessout",JSON.stringify(chess.pgn()));
}
thegameofchess=[fxchess()];
module.exports = thegameofchess;

/* COPILOT EXPLAIN
This JavaScript file, `the_random_game_of_chess.js`, simulates a random game of chess and logs the result. Here's a breakdown of what it does:

1. **Require Modules**:
   - `openvino-node`: Required but not used in the code.
   - `chess.js`: A library to handle chess logic.
   - `fs/promises`: The file system module for handling file operations asynchronously.

2. **Initialize Array**:
   - `thegameofchess`: An empty array intended to store the result of the chess game.

3. **Async Function `fxchess`**:
   - **Create Chess Game**: Initializes a new chess game using the `Chess` class from `chess.js`.
   - **Simulate Moves**: Continuously generates random legal moves until the game is over.
   - **Log Game Result**: Appends the game's PGN (Portable Game Notation) to a file named `chessout`.

4. **Execute Function**:
   - Executes the `fxchess` function and stores the returned promise in the `thegameofchess` array.

5. **Export Module**:
   - Exports the `thegameofchess` array as a module.

This script is used to simulate a random game of chess and save the game's PGN to a file.

*/




