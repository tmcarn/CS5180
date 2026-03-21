import numpy as np

def check_valid_move(i, j):
    if i < 0 or i >= four_room_space.shape[0] or j < 0 or j >= four_room_space.shape[1]:
        return False # Out of bounds
    
    elif four_room_space[i, j] == 1:
        return False # Wall
    
    return True

four_room_space = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])

count = 0

for i in range(four_room_space.shape[0]):
    for j in range(four_room_space.shape[1]):
        if four_room_space[i, j] != 1: # Skip walls
            # Iterate over possible moves (up, down, left, right)
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    if abs(di) != abs(dj): # Ignore diagonal moves and no-moves
                        # Check how many possible next states there are
                        ni, nj = i + di, j + dj
                        if check_valid_move(ni, nj):
                            count += 1

print("Total valid moves in the four-room environment:", count)

