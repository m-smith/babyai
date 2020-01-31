import numpy as np


def log_uniform(low=1e-9, high=1, size=None):
    if low <= 0:
        low = np.finfo("float").eps
    return np.exp(np.random.uniform(np.log(low), np.log(high), size=size))


class ArgPreprocessor(object):
    """Samples hyperparameters randomly rather than as a grid"""

    def __init__(self, args, preprocessor):
        self.args = args
        self.preprocessor = preprocessor

    def __call__(self):
        try:
            return self.preprocessor(**self.args)
        except TypeError as e:
            if "**" in str(e):
                try:
                    return self.preprocessor(*self.args)
                except TypeError as e:
                    if "*" in str(e):
                        return self.preprocessor(self.args)
                    raise
            raise


class RandomParameter(ArgPreprocessor):
    """Samples hyperparameters randomly rather than as a grid"""

    def __init__(self, params, dist):
        super().__init__(params, dist)


class BooleanParameter(object):
    pass
