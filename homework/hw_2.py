# Code to generate table
import random
import numpy as np
import tqdm.notebook as tqdm
import matplotlib.pyplot as plt
import ipywidgets as widgets
from tqdm import tqdm


# FOUR ROOM ENVIRONMENT
class FourRooms(object):
    def __init__(self,seed = 2025):
        # define the four room as a 2-D array for easy state space reference and visualization
        # 0 represents an empty cell; 1 represents a wall cell
        self.four_room_space = np.array([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
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
        
        # find the positions for all empty cells
        # not that: the origin for a 2-D numpy array is located at top-left while the origin for the FourRooms is at
        # the bottom-left. The following codes performs the re-projection.
        empty_cells = np.where(self.four_room_space == 0.0)
        self.state_space = [[col, 10 - row] for row, col in zip(empty_cells[0], empty_cells[1])]

        # define the action space
        self.action_space = {'LEFT': np.array([-1, 0]),
                             'RIGHT': np.array([1, 0]),
                             'DOWN': np.array([0, -1]),
                             'UP': np.array([0, 1])}
        
        # define the start state
        self.start_state = [0, 0]
        
        # define the goal state
        self.goal_state = [10, 10]


        # Action setup
        self.nprng = np.random.default_rng(seed=seed)
        self.possible_results = ["success","left","right"]
        self.probs = [0.8, 0.1, 0.1]

        self.left_map = {"UP":"LEFT", "LEFT":"DOWN", "DOWN":"RIGHT", "RIGHT":"UP"}
        self.right_map = {"UP":"RIGHT", "RIGHT":"DOWN", "DOWN":"LEFT", "LEFT":"UP"}
        

    def reset(self):
        """
        Reset the agent's state to the start state [0, 0]
        Return both the start state and reward
        """
        state = self.start_state  # reset the agent to [0, 0]
        reward = 0  # reward is 0
        return state, reward
        

    def step(self, state, act):
        """
        Args: 
            state: a list variable containing x, y integer coordinates. (i.e., [1, 1]).
            act: a string variable (i.e., "UP"). All feasible values are ["UP", "DOWN", "LEFT", "RIGHT"].
        Output args: 
            next_state: a list variable containing x, y integer coordinates (i.e., [1, 1])
            reward: an integer. it can be either 0 or 1.
        """
        
        # CODE HERE: implement the stochastic dynamics as described in Q1. 
        # Please note, we provide you with the deterministic transition function "take_action" below.
        # Therefore, you only have to implement the logics of the stochasticity.
        
        # Determine success/slippage of action
        result = self.nprng.choice(self.possible_results, p=self.probs)

        # Process action based on result
        if result == "success":
            next_state = self.take_action(state,act)
        elif result == "left":
            next_state = self.take_action(state,self.left_map[act])
            #print("slipped left")
        elif result == "right":
            next_state = self.take_action(state,self.right_map[act])
            #print("slipped right")
        else:
            next_state = state
            print("Something is wrong")
        

        # CODE HERE: compute the reward based on the resulting state
        # Reward reset is handled in testing framework not the environment
        if next_state == self.goal_state:
            reward = 1
        else:
            reward = 0
        

        # return the current state, reward
        return next_state, reward
        

    """ DO NOT CHANGE BELOW """
    def take_action(self, state, act):
        """
        Input args: 
            state (list): a list variable containing x, y integer coordinates. (i.e., [1, 1]).
            act (string): a string variable (i.e., "UP"). All feasible values are ["UP", "DOWN", "LEFT", "RIGHT"].
        Output args: 
            next_state (list): a list variable containing x, y integer coordinates (i.e., [1, 1])
        """
        state = np.array(state)
        next_state = state + self.action_space[act]
        return next_state.tolist() if next_state.tolist() in self.state_space else state.tolist()
    



if __name__=="__main__":
    world = FourRooms()

    outpath = "./hw2_table.txt"

    # next_state = world.take_action([0,0],"UP")
    # print(next_state)

    with open(outpath,"w") as file:
        file.write("State , Action | Next State, Reward | Probability\n")
        for state in world.state_space:
            if state == world.goal_state:
                reward = 1
                for action in world.action_space:
                    next_state = world.start_state
                    prob = 1
                    file.write(f"[{int(state[0]):2} {int(state[1]):2}], {action:<5} | [{int(next_state[0]):2} {int(next_state[1]):2}], {reward:1} | {prob}\n")
            else:
                reward = 0
                
                for action in world.action_space:
                    for idx,next_state in enumerate([world.take_action(state,action),world.take_action(state,world.left_map[action]),world.take_action(state,world.right_map[action])]):
                        file.write(f"[{int(state[0]):2} {int(state[1]):2}], {action:<5} | [{int(next_state[0]):2} {int(next_state[1]):2}], {reward:1} | {world.probs[idx]:3}\n")


    print(f"Table written to {outpath}")