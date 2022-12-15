
import torch
import numpy as np


class PracticeClass:

    def f(self, x: torch.Tensor = None, verbose: bool = False) -> torch.float:


        return x + 1

    def g(self,x: torch.Tensor = None, verbose: bool = False) -> torch.float:
        """replicating the function fu in order to check if the code is working correctly."""

        return x ** 2

        
    def Fs(self, x):
        return [self.f(x), self.g(x)]

    def Fss(self):
        return np.array([self.f, self.g])