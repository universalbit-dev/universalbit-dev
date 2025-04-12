/*
 * mpi_random_game_of_chess.cpp
 *
 * Parallel Chess Game Simulation
 * 
 * This program simulates random chess games using MPI for distributed computing.
 * Each process simulates a portion of the games, computes the results, and gathers 
 * the results at the root process.
 *
 * Compilation:
 *   mpic++ -o mpi_random_game_of_chess mpi_random_game_of_chess.cpp
 *
 * Execution:
 *   mpirun -np <num_processes> ./mpi_random_game_of_chess
 *   Replace <num_processes> with the number of processes to use.
 *
 * Example:
 *   mpirun -np 4 ./mpi_random_game_of_chess
 *
 * Output:
 *   The program prints the results of the simulated chess games, such as which player won
 *   (White or Black), or if the game was a Draw, for each game simulated.
 *
 * Repository: https://github.com/universalbit-dev/
 * Date: April 2025
*/


#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>

// Constants
#define NUM_GAMES 20
#define BOARD_SIZE 64
#define MAX_MOVES 100
#define WHITE_PIECE 1
#define BLACK_PIECE -1
#define EMPTY 0

// Function prototypes
void initChessBoards(std::vector<int>& boards, int numGames);
void simulateRandomGames(std::vector<int>& boards, std::vector<int>& results, int boardSize, int maxMoves);
void gatherAndPrintResults(int rank, int size, const std::vector<int>& localResults, int gamesPerProcess);

int main(int argc, char** argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Determine the number of games per process
    int gamesPerProcess = NUM_GAMES / size;

    std::vector<int> boards(gamesPerProcess * BOARD_SIZE, EMPTY);
    std::vector<int> results(gamesPerProcess, 0);

    // Initialize boards on rank 0 and scatter to all processes
    if (rank == 0) {
        std::vector<int> allBoards(NUM_GAMES * BOARD_SIZE, EMPTY);
        initChessBoards(allBoards, NUM_GAMES);
        MPI_Scatter(allBoards.data(), gamesPerProcess * BOARD_SIZE, MPI_INT, boards.data(), 
                    gamesPerProcess * BOARD_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        MPI_Scatter(nullptr, gamesPerProcess * BOARD_SIZE, MPI_INT, boards.data(), 
                    gamesPerProcess * BOARD_SIZE, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Simulate random chess games
    simulateRandomGames(boards, results, BOARD_SIZE, MAX_MOVES);

    // Gather results at rank 0
    if (rank == 0) {
        std::vector<int> allResults(NUM_GAMES, 0);
        MPI_Gather(results.data(), gamesPerProcess, MPI_INT, allResults.data(), 
                   gamesPerProcess, MPI_INT, 0, MPI_COMM_WORLD);
        gatherAndPrintResults(rank, size, allResults, gamesPerProcess);
    } else {
        MPI_Gather(results.data(), gamesPerProcess, MPI_INT, nullptr, gamesPerProcess, MPI_INT, 0, MPI_COMM_WORLD);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

// Initialize chess boards with default positions
void initChessBoards(std::vector<int>& boards, int numGames) {
    for (int game = 0; game < numGames; game++) {
        for (int i = 0; i < BOARD_SIZE; i++) {
            boards[game * BOARD_SIZE + i] = EMPTY; // Empty board
        }
        for (int i = 0; i < 8; i++) {
            boards[game * BOARD_SIZE + 8 + i] = WHITE_PIECE; // White pawns
            boards[game * BOARD_SIZE + 48 + i] = BLACK_PIECE; // Black pawns
        }
    }
}

// Simulate random chess games
void simulateRandomGames(std::vector<int>& boards, std::vector<int>& results, int boardSize, int maxMoves) {
    srand(time(nullptr) + MPI_Wtime()); // Seed random number generator

    for (size_t game = 0; game < results.size(); game++) {
        int* board = &boards[game * boardSize];
        int moveCount = 0;
        bool whiteTurn = true; // Alternate turns between white and black

        while (moveCount < maxMoves) {
            // Random moves (dummy simulation)
            int move = rand() % boardSize;
            if (board[move] == EMPTY) {
                board[move] = whiteTurn ? WHITE_PIECE : BLACK_PIECE;
                whiteTurn = !whiteTurn;
                moveCount++;
            }
        }

        // Randomly assign a result
        results[game] = rand() % 3 - 1; // -1 (Black wins), 0 (Draw), 1 (White wins)
    }
}

// Gather and print results
void gatherAndPrintResults(int rank, int size, const std::vector<int>& results, int gamesPerProcess) {
    if (rank == 0) {
        std::cout << "Simulating " << NUM_GAMES << " chess games with MPI...\n";
        for (size_t i = 0; i < results.size(); i++) {
            std::cout << "Game " << i + 1 << ": ";
            if (results[i] == 1) {
                std::cout << "White wins!\n";
            } else if (results[i] == -1) {
                std::cout << "Black wins!\n";
            } else if (results[i] == 0) {
                std::cout << "Draw!\n";
            }
        }
    }
}
