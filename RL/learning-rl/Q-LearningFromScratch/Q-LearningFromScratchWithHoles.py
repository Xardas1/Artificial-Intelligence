import numpy as np
import pdb

"""
Q-Learning from Scratch with Holes Environment
============================================

This script implements a Q-learning algorithm from scratch to train an agent to navigate
a 5x5 grid world with holes (obstacles) and reach a goal.

Environment:
- 5x5 grid with agent (A), goal (G), holes (H), and empty spaces (-)
- Agent starts at top-left (0,0), goal is at bottom-right (4,4)
- Two holes at positions (0,4) and (1,2) that give -100 reward
- All other moves give -0.1 reward, reaching goal gives +1 reward

Actions:
- 0: Right, 1: Left, 2: Down, 3: Up

Q-Learning Parameters:
- Learning rate (alpha): 0.01 for training, 0.1 for manual
- Discount factor (gamma): 0.99
- Epsilon: starts at 1, decays by factor 0.999 each episode
- Training episodes: 50,000
"""

def grid_prepare(grid_size):
    """
    Initialize a grid with empty spaces.
    
    Args:
        grid_size (int): Size of the grid (5x5)
    
    Returns:
        list: 2D list representing the grid with '-' as empty spaces
    """
    grid = [['-' for _ in range(grid_size)] for _ in range(5)]
    return grid

def grid_create(prepared_grid):
    """
    Convert the 2D grid to a printable string format.
    
    Args:
        prepared_grid (list): 2D list representing the grid
    
    Returns:
        str: String representation of the grid for display
    """
    grided_grid = ["".join(k) for k in prepared_grid]
    fixed_grid = "\n".join(grided_grid)
    return fixed_grid

def agent_movement(direction, agent_row, agent_column):
    """
    Move the agent in the specified direction if valid.
    
    Args:
        direction (str): Direction of movement ('Right', 'Left', 'Down', 'Up')
        agent_row (int): Current row position
        agent_column (int): Current column position
    
    Returns:
        tuple: (updated_grid, new_row, new_column)
    """ 
    if direction == "Right" and agent_column < 4    :
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
        direction_moved = "Wrong Direction"
    return prepared_grid, agent_row, agent_column

def future_state(agent_row, agent_column, agent_row_fut, agent_column_fut, action):
    """
    Calculate the future state position based on the action.
    
    Args:
        agent_row (int): Current row position
        agent_column (int): Current column position
        agent_row_fut (int): Future row position (placeholder)
        agent_column_fut (int): Future column position (placeholder)
        action (int): Action number (0=Right, 1=Left, 2=Down, 3=Up)
    
    Returns:
        tuple: (future_row, future_column)
    """
    if action == 0 and agent_column < 4:
        agent_column_fut = agent_column + 1
    elif action == 1 and agent_column > 0:
        agent_column_fut = agent_column - 1
    elif action == 2 and agent_row < 4:
        agent_row_fut = agent_row + 1
    elif action == 3 and agent_row > 0:
        agent_row_fut = agent_row - 1
    return agent_row_fut, agent_column_fut

def map_grid_to_states(prepared_grid):
    """
    Map grid positions to state numbers for Q-table indexing.
    
    Args:
        prepared_grid (list): 2D grid representation
    
    Returns:
        list: Alternating list of (row, col) tuples and state numbers
    """
    state_list = []
    a = -1
    for row in range(len(prepared_grid)):
        for col in range(len(prepared_grid)):
            a += 1
            state_list.append((row, col))
            state_list.append(a)
    return state_list

def count_states(state_list):
    """
    Create a dictionary mapping grid positions to state numbers.
    
    Args:
        state_list (list): List from map_grid_to_states function
    
    Returns:
        dict: Dictionary with (row, col) keys and state number values
    """
    state_counted = dict(zip(state_list[0::2], state_list[1::2]))
    return state_counted

def visual_grid_prep(agent_row, agent_column):
    prepared_grid[0][0] = 'A'
    prepared_grid[4][4] = 'G'
    grid = grid_create(prepared_grid)
    print(grid)
    return agent_row, agent_column

def number_of_steps(step_number):
    end_of_episode = False
    if step_number == 24:
        end_of_episode = True
        return end_of_episode
        
def creating_Qtable():
    """
    Initialize the Q-table and action mapping.
    
    Returns:
        tuple: (Q_table, movement_dict)
            Q_table: 25x4 numpy array (25 states, 4 actions)
            movement_dict: Dictionary mapping action numbers to direction names
    """
    Q_table = np.zeros([25, 4])
    movment_list = {0 : 'Right', 1 : 'Left', 2 : 'Down', 3 : 'Up'}
    return Q_table, movment_list

def epsilon_decaying_greedy_algorithm(epsilon, Q_table, state_number, number_actions):
    if np.random.random() > epsilon:
        action = np.argmax(Q_table[state_number])
    else:
        action = np.random.choice(number_actions)
    return action


def Ai_handling_training(movment_list, agent_row, agent_column, Q_table, reward_list, state_list, epsilon):
    list_of_actions = [0,1,2,3]
    step_number = 0
    while agent_row != 4 or agent_column != 4:
        state_number = state_list[(agent_row , agent_column)]                               
        action_number = epsilon_decaying_greedy_algorithm(epsilon, Q_table, state_number, list_of_actions)
        agent_row_fut, agent_column_fut = future_state(agent_row, agent_column, agent_row, agent_column, action_number)
        maxQ_calculated = maxQ(agent_row_fut, agent_column_fut, reward_list)
        Q_table = Q_learning_algorithm(state_number, agent_row_fut, agent_column_fut, Q_table, action_number, maxQ_calculated, 0.01, 0.99)
        if agent_row_fut == 1 and agent_column_fut == 2:
            break
        if agent_row_fut == 0 and agent_column_fut == 4:
            break
      
        action = movment_list[action_number]
        prepared_grid, agent_row, agent_column = agent_movement(action, agent_row, agent_column)
        step_number += 1
        grid = grid_create(prepared_grid)
       
        end_of_episode = number_of_steps(step_number)
        if end_of_episode == True:
            return Q_table
        elif agent_row == 4 and agent_column == 4:
    
            return Q_table
           
    return Q_table

def maxQ(agent_row_fut, agent_column_fut, reward_list):
    optimal_Q = []
    if agent_column_fut < 4:
        optimal_Q.append(reward_list[agent_row_fut][agent_column_fut+1])
    if agent_column_fut > 0:
        optimal_Q.append(reward_list[agent_row_fut][agent_column_fut-1])
    if agent_row_fut < 4:
        optimal_Q.append(reward_list[agent_row_fut+1][agent_column_fut])
    if agent_row_fut > 0:
        optimal_Q.append(reward_list[agent_row_fut-1][agent_column_fut])
    return np.max(optimal_Q)

def Q_learning_algorithm(state_number,  agent_row_fut, agent_column_fut, Q_table, action, max_Q, alfa, gamma):
    if agent_row_fut == 4 and agent_column_fut == 4:
        reward = 1
    elif agent_row_fut == 1 and agent_column_fut == 2:
        reward = -100
    elif agent_row_fut == 0 and agent_column_fut == 4:
        reward = -100
    else:
        reward = -0.1
    Q_table[state_number][action] = Q_table[state_number][action] + alfa * (reward + gamma * max_Q - Q_table[state_number][action])
    return Q_table

def create_optimal_action(Q_table):
    optimal_actions = []
    for i in Q_table:
        optimal_actions.append(np.argmax(i))
    return optimal_actions

def trained_ai(agent_row, agent_column, movement_list, optimal_actions):
    step_number = 0
    n = 0
    while agent_row != 4 or agent_column != 4:
        action = movement_list[optimal_actions[n]]
        prepared_grid, agent_row, agent_column = agent_movement(action, agent_row, agent_column)
        prepared_grid[0][4] = "H"
        prepared_grid[1][2] = "H"
        grid = grid_create(prepared_grid)
        print(grid)
        step_number += 1
        if step_number > 24:
            break
        else:
            n += 1
        print("Number of Step: ", step_number)
        end_of_episode = number_of_steps(step_number)
        if end_of_episode == True:
            print("End of Episode")
            print("+0 Reward")
        elif agent_row == 4 and agent_column == 4:
            print("End of Episode")
            print("+1 Reward")


reward_list = [[-0.2, -0.2, -0.2, -0.2, -100],
               [-0.2, -0.2, -100, -0.1, 0.0],
               [-0.2, -0.2, -0.1, 0.0, 0.2],
               [-0.2, -0.1, 0.0, 0.2, 0.5],
               [-0.1,  0.0, 0.2, 0.5, 1]]

reward_list_test = [[0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 0],
                   [0, 0, 0, 0, 1]]

prepared_grid = grid_prepare(5)

maped_grid = map_grid_to_states(prepared_grid)

counted_states_dict = count_states(maped_grid)


Q_table, movement_list = creating_Qtable()



a = 0
epsilon = 1


while a < 50000:
    prepared_grid = grid_prepare(5)
    Q_table = Ai_handling_training(movement_list, 0, 0, Q_table, reward_list, counted_states_dict, epsilon)

    epsilon = epsilon * 0.999
    a += 1


optimal_actions = create_optimal_action(Q_table)



prepared_grid = grid_prepare(5)
trained_ai(0, 0, movement_list, optimal_actions)

