import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, state_dims, action_dims, bd_dims):
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, state_dims),
                                dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, state_dims),
                                dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, action_dims),
                                dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.bds_memory = np.zeros((self.mem_size, bd_dims), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, bds):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = int(done)
        self.bds_memory[index] = bds
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]
        bds = self.bds_memory[batch]

        return states, actions, rewards, states_, terminal, bds
