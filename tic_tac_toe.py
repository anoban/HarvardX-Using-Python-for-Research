import random
import numpy as np
import matplotlib.pyplot as plt
import os
import gc
random.seed(1)

# create board
def create_board():
    global board
    board = np.zeros((3,3), dtype=int)

create_board()

# check available coordinates
def possibilities(board):
    indices = np.where(board==0)  # nested list of x and y coordinates
    coordinates = []
    for c in range(0,(len(indices[0]))):
        coordinates.append((indices[0][c],indices[1][c]))
    return coordinates

possibilities(board)

# randomly placing a marker on available positions
def random_place(board,player):
    board[random.choice(possibilities(board))] = int(player)
    return board

# two players each placing 3 markers alternatively
for player in (1,2,1,2,1,2):
    random_place(board, player)

board
# array([[2, 2, 1],
#       [0, 1, 0],
#       [0, 1, 2]])

# check has anyone won row wise
def row_win(board, player):
    check_rows = np.all(board == player, axis=1)
    if np.any(check_rows) == True:
        return player
    else:
        pass

# check has anyone won column wise
def col_win(board, player):
    check_cols = np.all(board == player, axis=0)
    if np.any(check_cols) == True:
        return player
    else:
        pass

# check has anyone won diagonally
def diag_win(board, player):
    check_diag1 = np.all(board.diagonal() == player)
    check_diag2 = np.all(np.fliplr(board).diagonal() == player)
    diagonals = np.array([check_diag1, check_diag2])
    if np.any(diagonals) == True:
        return player
    else:
        pass

# Create a function evaluate(board) that uses row_win, col_win, and diag_win 
# functions for both players. If one of them has won, 
# return that player's number. 
# If the board is full but no one has won, return -1. Otherwise, return 0

# return statements can only be placed inside function definitions. 
# You cannot return a value from a standalone loop

# There are two solutions to this sort of problem. 
# One is to replace return with a break statement, which will exit the loop at the specified condition
# other is to wrap the return inside a function

def evaluate(board):
    for player in range(1,3):
        if np.any(np.array([row_win(board, player), col_win(board, player), diag_win(board, player)]) == player) == True:
            return player
            break
            if len(possibilities(board)) == 0 and player != 0:
                return -1
            else:
                return 0
    
            
evaluate(board)

# looped game
def play_game():
    create_board()  # creates a 3x3 empty board
    while evaluate(board) == 0:  # while no one has won yet 
        for player in (1,2):
            if len(possibilities(board)) != 0:  # if the board isn't filled yet
                random_place(board, player)
    print(evaluate(board))
    return board

# 1000 repetitions
results = []
for i in range(0,1000):
    results.append(play_game())

