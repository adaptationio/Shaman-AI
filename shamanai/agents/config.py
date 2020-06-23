from ..env import BattleAgents


class Config():
    def __init__(self):
        self.env = BattleAgents
        self.name = "Vladamir"
        self.agent = "PPO"
        self.policy = "Mlp"
        self.eval = False
        self.tensorboard = "shit"
        self.save = "saves/save"
        self.start = 0
        self.end = 0
        self.steps = 0
        self.random = False
        self.sequence = False
        self.os = 0
        self.norm = 0
        self.best = 0
        self.params = {
            'cliprange': 0.2750880341027778,
            'ent_coef': 0.0012029768441485513,
            'gamma': 0.977036879481983,
            'lam': 0.9845620727331588,
            'learning_rate': 0.0001411368451162598,
            'n_steps': int(374),
            'noptepochs': int(37)
            }
        self.sl = 0
