import numpy as np
import pdb

def grid_prepare(grid_size):
    grid = [['-' for _ in range(grid+siz)] for _ in range(5)]
    return grid



def grid_create(prepared_grid):
    grided_grid = ["".join(k) for k in prepared_grid]
    fixed_grid = "\n".join(grided_grid)
    return fixed_grid

def agent_movement(direction, agent_row, agent_column): 
    if direction == "Right" and agent_column < 4    :
        prepared_grid[agent_row][agent_column] = '-'
        agent_column += 1
        prepared_grid[agent_row][agent_column] = 'A'
        #print(direction_moved)
    elif direction == "Left" and agent_column > 0:
        prepared_grid[agent_row][agent_column] = '-'
        agent_column -= 1
        prepared_grid[agent_row][agent_column] = 'A'
        #print(direction_moved)
    elif direction == "Down" and agent_row < 4:
        prepared_grid[agent_row][agent_column] = '-'
        agent_row += 1
        prepared_grid[agent_row][agent_column] = 'A'
        #print(direction_moved)
    elif direction == "Up" and agent_row > 0:
        prepared_grid[agent_row][agent_column] = '-'
        agent_row -= 1
        prepared_grid[agent_row][agent_column] = 'A'
        #print(direction_moved)
    else:
        #print("Wrong Direction, try again")
        direction_moved = "Wrong Direction"
    return prepared_grid, agent_row, agent_column

def future_state(agent_row, agent_column, agent_row_fut, agent_column_fut, action):
    if action == 0 and agent_column < 4:
        agent_column_fut = agent_column + 1
    elif action == 1 and agent_column_fut > 0:
        agent_column_fut = agent_column - 1
    elif action == 2 and agent_row < 4:
        agent_row_fut = agent_row + 1
    elif action == 3 and agent_row > 0:
        agent_row_fut = agent_row - 1
    return agent_row_fut, agent_column_fut

def map_grid_to_states(prepared_grid):
    state_list = []
    a = -1
    for row in range(len(prepared_grid)):
        for col in range(len(prepared_grid)):
            a += 1
            state_list.append((row, col))
            state_list.append(a)
    return state_list

def count_states(state_list):
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
        #print("+0 Reward")
        #print("End of Episode")
        end_of_episode = True
        return end_of_episode
        
def creating_Qtable():
    Q_table = np.zeros([25, 4])
    movment_list = {0 : 'Right', 1 : 'Left', 2 : 'Down', 3 : 'Up'}
    return Q_table, movment_list

def epsilon_decaying_greedy_algorithm(epsilon, Q_table, state_number, number_actions):
    if np.random.random() > epsilon:
        action = np.argmax(Q_table[state_number])
    else:
        action = np.random.choice(number_actions)
    return action

def manual_handling(agent_row, agent_column, state_list, Q_table):
    step_number = 0
    while agent_row != 4 or agent_column != 4:
        state_number = state_list[(agent_row, agent_column)]
        action_number = int(input("Give direction: "))
        agent_row_fut, agent_column_fut = future_state(agent_row, agent_column, agent_row, agent_column, action_number)
        maxQ_calculated = maxQ(agent_row_fut, agent_column_fut, reward_list)
        action = movement_list[action_number]
        Q_table = Q_learning_algorithm(state_number, agent_row_fut, agent_column_fut, Q_table, action_number, maxQ_calculated, 0.1, 0.99)
        if agent_row_fut == 1 and agent_column_fut == 2:
            print("Death")
            print("Reward -100")
            break
        elif agent_row_fut == 0 and agent_column_fut == 4:
            print("Death")
            print("Reward -100")
            break
        prepared_grid, agent_row, agent_column = agent_movement(action, agent_row, agent_column)
        prepared_grid[0][4] = "H"
        prepared_grid[1][2] = "H"
        grid = grid_create(prepared_grid)
        print(grid)
        step_number += 1
        print("Number of Step: " , step_number)
        end_of_episode = number_of_steps(step_number)
        if end_of_episode == True:
            print("End")
            return Q_table
        elif agent_row == 4 and agent_column == 4:
            print("+1 Reward")
            print("End of Episode")
            return Q_table
    return Q_table

def Ai_handling_training(movment_list, agent_row, agent_column, Q_table, reward_list, state_list, epsilon):
    list_of_actions = [0,1,2,3]
    step_number = 0
    while agent_row != 4 or agent_column != 4:
        state_number = state_list[(agent_row , agent_column)]                               
        action_number = epsilon_decaying_greedy_algorithm(epsilon, Q_table, state_number, list_of_actions)
        agent_row_fut, agent_column_fut = future_state(agent_row, agent_column, agent_row, agent_column, action_number)
        maxQ_calculated = maxQ(agent_row_fut, agent_column_fut, reward_list)
        Q_table = Q_learning_algorithm(state_number, agent_row_fut, agent_column_fut, Q_table, action_number, maxQ_calculated, 0.01, 0.99)
        if agent_row_fut == 1 and agent_column_fut == 4:
            break
        if agent_row_fut == 0 and agent_column_fut == 4:
            break
        #print("State number : " , state_number)
        #print("Action number : ", action_number)
        #print("Agent row : ", agent_row, "Agent column", agent_column)
        #print("Agent row fut : ", agent_row_fut, "Agent col fut" , agent_column_fut)
        #print("Reward List : ", reward_list)
        action = movment_list[action_number]
        prepared_grid, agent_row, agent_column = agent_movement(action, agent_row, agent_column)
        step_number += 1
        grid = grid_create(prepared_grid)
        #print(grid)
        #print("Number of Step: ", step_number)
        end_of_episode = number_of_steps(step_number)
        #pdb.set_trace()
        if end_of_episode == True:
            return Q_table
        elif agent_row == 4 and agent_column == 4:
            #print("+1 Reward")
            #print("End of Episode")
            return Q_table
           
    return Q_table

def maxQ(agent_row_fut, agent_column_fut, reward_list):
    optimal_Q = []
    if agent_column_fut < 4:
        optimal_Q.append(reward_list[agent_row_fut][agent_column_fut+1])
    if agent_column_fut > 0:
        optimal_Q.append(reward_list[agent_row_fut][agent_column_fut-1])
        #pdb.set_trace()
    if agent_row_fut < 4:
        optimal_Q.append(reward_list[agent_row_fut+1][agent_column_fut])
        #pdb.set_trace()
    if agent_row_fut > 0:
        optimal_Q.append(reward_list[agent_row_fut-1][agent_column_fut])
        #pdb.set_trace()
    #pdb.set_trace()
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
    #pdb.set_trace()
    Q_table[state_number][action] = Q_table[state_number][action] + alfa * (reward + gamma * max_Q - Q_table[state_number][action])
    return Q_table

def create_optimal_action():
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

prepared_grid = grid_prepare()

maped_grid = map_grid_to_states(prepared_grid)

counted_states_dict = count_states(maped_grid)


Q_table, movement_list = creating_Qtable()



a = 0
epsilon = 1


while a < 50000:
    prepared_grid = grid_prepare()
    Q_table = Ai_handling_training(movement_list, 0, 0, Q_table, reward_list, counted_states_dict, epsilon)

    epsilon = epsilon * 0.999
    a += 1


optimal_actions = create_optimal_action()


#manual_handling(0, 0, counted_states_dict, Q_table)

#print(Q_table)
prepared_grid = grid_prepare()
trained_ai(0, 0, movement_list, optimal_actions)

