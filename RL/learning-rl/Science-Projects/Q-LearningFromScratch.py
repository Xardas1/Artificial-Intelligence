import numpy as np

def grid_prepare():
    grid = [['-' for _ in range(5)] for _ in range(5)]
    return grid

def grid_create(prepared_grid):
    grided_grid = ["".join(k) for k in prepared_grid]
    fixed_grid = "\n".join(grided_grid)
    return fixed_grid

def agent_movement(direction, agent_row, agent_column):
    if direction == "Right" and agent_column < 4:
        prepared_grid[agent_row][agent_column] = '-'
        agent_column += 1
        prepared_grid[agent_row][agent_column] = 'A'
    elif direction == "Left" and agent_column > 0:
        prepared_grid[agent_row][agent_column] = '-'
        agent_column -= 1
        prepared_grid[agent_row][agent_column] = 'A'
    elif direction == "Down" and agent_row < 4:
        prepared_grid[agent_row][agent_column] = '-'
        agent_row += 1
        prepared_grid[agent_row][agent_column] = 'A'
    elif direction == "Up" and agent_row > 0:
        prepared_grid[agent_row][agent_column] = '-'
        agent_row -= 1
        prepared_grid[agent_row][agent_column] = 'A'
    else:
        print("Wrong Direction, try again")
    return prepared_grid, agent_row, agent_column

prepared_grid = grid_prepare()

def visual_grid_prep(agent_row, agent_column):
    prepared_grid[0][0] = 'A'
    prepared_grid[4][4] = 'G'
    grid = grid_create(prepared_grid)
    print(grid)
    return agent_row, agent_column


agent_row, agent_column = visual_grid_prep(0, 0)

def number_of_steps(step_number):
    if step_number == 1000:
        print("+0 Reward")
        print("End of Episode")
        return True

def manual_handling(agent_row, agent_column):
    step_number = 0
    while agent_row != 4 or agent_column != 4:
        direction = input("Give direction: ")
        prepared_grid, agent_row, agent_column = agent_movement(direction, agent_row, agent_column)
        step_number += 1
        grid = grid_create(prepared_grid)
        print(grid)
        print("Number of Step: " , step_number)
        if number_of_steps(step_number) == True:
            break
        elif agent_row == 4 and agent_column == 4:
            print("+1 Reward")
            print("End of Episode")


manual_handling(agent_row, agent_column)


