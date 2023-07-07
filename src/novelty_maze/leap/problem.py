from leap_ec.problem import ScalarProblem
from ..env import MazeEnvironment, HARD_MAZE


class MazeProblem(ScalarProblem):

    # From NoveltySearch
    max_timesteps = 400

    def __init__(self, maze_fp=HARD_MAZE, use_bias=False):
        super().__init__(True)
        self.env = MazeEnvironment(maze_fp)
        self.use_bias = use_bias
    
    def evaluate(self, phenome):
        num_collisions = 0
        self.env.reset()
        for _ in range(self.max_timesteps):
            obs = self.env.hero.get_observation()
            act = phenome(obs[1 - self.use_bias:]) # If use_bias is False, removes the bias input

            dist, collide, goal = self.env.update(act)
            num_collisions += collide
            if goal:
                break
        
        evaluation = {
                "distance": dist,
                "goal": goal,
                "final_pos": self.env.hero.location,
                "collisions": num_collisions
            }
        evaluation["quality"] = max(0.1, 300 - dist)
        evaluation["behavior"] = evaluation["final_pos"] / 100 - 1 # map final pos to -1, 1
        return evaluation