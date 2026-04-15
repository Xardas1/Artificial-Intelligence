# Q-Learning from Scratch with Holes Environment

Implementation of Q-learning algorithm to navigate a 5x5 grid with obstacles and reach a goal.

## Environment

- **Grid**: 5x5
- **Start**: (0,0) - Agent 'A'
- **Goal**: (4,4) - Goal 'G'
- **Holes**: (0,4) and (1,2) - Obstacles 'H'
- **Empty**: '-'

### Grid Layout
```
-A--H
--H--
-----
-----
----G
```

## Actions

- **0**: Right
- **1**: Left
- **2**: Down
- **3**: Up

## Rewards

- **Goal**: +1
- **Hole**: -100 (episode ends)
- **Move**: -0.1
- **Timeout**: 0 (24 steps)

## Algorithm

- **Q-Table**: 25x4 matrix (states × actions)
- **States**: Grid positions 0-24
- **Policy**: Epsilon-greedy with decay
- **Update**: Q(s,a) = Q(s,a) + ×[r + ×max(Q(s',a')) - Q(s,a)]

## Parameters

- **Learning Rate**: 0.01
- **Discount Factor**: 0.99
- **Epsilon**: 1.0 → 0.999 decay
- **Episodes**: 50,000

## Files

- `Q-LearningFromScratchWithHoles.py` - Main code
- `5x5 enviornment.png` - Grid visual
- `README.md` - Documentation

## Functions

**Grid:**
- `grid_prepare()` - Initialize grid
- `grid_create()` - Format for display
- `agent_movement()` - Move agent

**States:**
- `map_grid_to_states()` - Position to state mapping
- `count_states()` - State dictionary
- `future_state()` - Next position

**Q-Learning:**
- `creating_Qtable()` - Initialize Q-table
- `epsilon_decaying_greedy_algorithm()` - Action selection
- `Q_learning_algorithm()` - Update Q-values
- `maxQ()` - Max Q-value

**Training:**
- `Ai_handling_training()` - Training loop
- `trained_ai()` - Test trained agent
- `create_optimal_action()` - Extract policy

## Usage

```bash
python3 Q-LearningFromScratchWithHoles.py
```

## Output

```
-A--H
--H--
-----
-----
-----
Number of Step:  1
--A-H
--H--
-----
-----
-----
Number of Step:  2
...
End of Episode
+1 Reward
```

## Dependencies

- Python 3.x
- NumPy

## Learning Topics

- Reinforcement learning basics
- Q-learning implementation
- Grid world navigation
- Policy optimization
- Exploration strategies

## Performance

- More episodes = better convergence
- Tune epsilon decay
- Adjust rewards
