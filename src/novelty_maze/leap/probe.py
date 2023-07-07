import matplotlib.pyplot as plt
from leap_ec.ops import Operator


class Scatter2DBehavior(Operator):

    def __init__(self, ax, keep_old_points=True, all_behaviors=False):
        super().__init__()
        self.ax = ax
        self.keep_old_points = keep_old_points

        self.points = []

        self.all_behaviors = all_behaviors
    
    def __call__(self, population):
        if not self.keep_old_points:
            self.points.clear()
        
        if self.all_behaviors:
            for ind in population:
                self.points.append(ind.evaluation["behavior"])
        else:
            self.points = [ind.evaluation["behavior"] for ind in population]

        self.ax.clear()
        
        x, y = zip(*self.points)
        self.ax.scatter(x, y)

        self.ax.update_datalim([(-1, -1), (1, 1)])
        self.ax.autoscale()

        plt.pause(0.0001)

        return population
