from abc import ABC,abstractmethod

class environment(ABC):
    @abstractmethod
    def get_info(self):
        raise NotImplementedError
    abstractmethod
    def random_state(self):
        raise NotImplementedError
    abstractmethod
    def step(self, state, actions, dt=1.0):
        # return new_state, is_end_state, reward
        raise NotImplementedError
