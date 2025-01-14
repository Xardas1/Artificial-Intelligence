from abc import ABC,abstractmethod

class algorithm(ABC):
    @abstractmethod
    def get_info(self):
        raise NotImplementedError
    abstractmethod
    def random_init(self,state_space_size,action_space_size,**kwargs):
        raise NotImplementedError
    def get_actions(self,state):
        raise NotImplementedError
    abstractmethod
    def apply_reward(self, samples, reward):
        raise NotImplementedError
    abstractmethod
    def train(self, samples):
        raise NotImplementedError
