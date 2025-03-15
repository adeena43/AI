import copy

# Define a simple 8x8 chess board (P: Pawn, N: Knight, B: Bishop, R: Rook, Q: Queen, K: King)
# Lowercase for black, uppercase for white
INITIAL_BOARD = [
    ["R", "N", "B", "Q", "K", "B", "N", "R"],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    [".", ".", ".", ".", ".", ".", ".", "."],
    ["p", "p", "p", "p", "p", "p", "p", "p"],
    ["r", "n", "b", "q", "k", "b", "n", "r"],
]

# Piece values for simple evaluation
PIECE_VALUES = {
    "P": 1, "N": 3, "B": 3, "R": 5, "Q": 9, "K": 100,
    "p": -1, "n": -3, "b": -3, "r": -5, "q": -9, "k": -100
}

def evaluate_board(board):
    """
    Evaluates the board by summing up piece values.
    A positive score means White is winning, a negative score means Black is winning.
    """
    score = 0
    for row in board:
        for piece in row:
            if piece in PIECE_VALUES:
                score += PIECE_VALUES[piece]
    return score

def generate_moves(board, is_white):
    """
    Generates simple pawn moves (for demonstration).
    This function should be expanded to handle other pieces.
    """
    moves = []
    direction = -1 if is_white else 1  # White moves up (-1), Black moves down (+1)

    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if (is_white and piece == "P") or (not is_white and piece == "p"):
                new_row = row + direction
                if 0 <= new_row < 8 and board[new_row][col] == ".":  # Forward move
                    new_board = copy.deepcopy(board)
                    new_board[new_row][col] = piece
                    new_board[row][col] = "."
                    moves.append((new_board, (row, col, new_row, col)))  # Store move details
    return moves

def beam_search(board, is_white, beam_width=3, depth_limit=2):
    """
    Beam Search to find the best move sequence.
    """
    beam = [(board, [], evaluate_board(board))]  # (board state, move sequence, score)

    for depth in range(depth_limit):
        candidates = []
        for current_board, move_sequence, score in beam:
            possible_moves = generate_moves(current_board, is_white)

            for new_board, move in possible_moves:
                new_score = evaluate_board(new_board)
                new_move_sequence = move_sequence + [move]
                candidates.append((new_board, new_move_sequence, new_score))

        # Sort by evaluation score and keep the best beam_width moves
        beam = sorted(candidates, key=lambda x: x[2], reverse=is_white)[:beam_width]

    # Return the best move sequence found
    return beam[0][1], beam[0][2]

# Run Beam Search
start_board = copy.deepcopy(INITIAL_BOARD)
is_white = True  # White to move
beam_width = 3
depth_limit = 2

best_moves, best_score = beam_search(start_board, is_white, beam_width, depth_limit)

# Print results
print("Best move sequence:")
for move in best_moves:
    print(f"Move: {move[0]}{move[1]} → {move[2]}{move[3]}")
print("Evaluation Score:", best_score)

