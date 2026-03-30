# 6x6 Checkers (PettingZoo AEC Environment)

This project implements a custom 6x6 Checkers environment using the PettingZoo AEC API, along with an Actor-Critic agent trained via self-play.

---

## Game Overview

- **Board**: 6x6 grid  
- **Players**: 2 (`player_0`, `player_1`)  
- **Movement**: Diagonal moves only  
- **Capturing**: Mandatory jumps  
- **King Rule**: Pieces become kings upon reaching the opposite end  
- **Win Condition**:
  - Capture all opponent pieces  
  - OR block opponent from having valid moves  

---

## Observation Space

The observation is a flattened vector of size **37**:

- First 36 values = board state (6×6 grid flattened)
  - `1` = player_0 piece  
  - `2` = player_0 king  
  - `-1` = player_1 piece  
  - `-2` = player_1 king  
  - `0` = empty cell 
- Last value = current player ID (0 or 1)  

This provides the agent with full information about the board.
The observation follows the PettingZoo AEC format and is provided to the current agent at each step.

---

## Action Space

- Discrete action space of size **144**
- Each action encodes:
  - piece position `(row, col)`
  - move direction (4 diagonal directions)

Only **valid moves** are allowed during training to ensure stability.

---

## Rewards

- **+1** = Win  
- **-1** = Loss  
- **+0.5** = Capture  
- **-0.2** = Invalid move  

Reward shaping is used to encourage capturing and strategic play.

---

## Termination Conditions

The game ends when:

- One player has **no remaining pieces**
- OR one player has **no legal moves available**

---

## Training (Actor-Critic with Self-Play)

- Actor selects actions using a probability distribution  
- Critic evaluates state quality  
- Both players share the same model  
- Training is done via self-play
- TD error is used for updates  