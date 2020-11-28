import numpy as np


class Evaluator(object):
    """Evaluator evaluates the model predictions by targets.
    """

    @classmethod
    def evaluate(cls, predictions, targets) -> dict:
        """evaluates predictions by targets.

        return a dict of evaluation results
        """
        raise NotImplementedError


class OneHotAccEvaluator(Evaluator):
    """OneHotAccEvaluator evaluates the one-hot encoded result.
    """

    @classmethod
    def evaluate(cls, predictions, targets) -> dict:
        predictions = np.argmax(predictions, axis=1)
        targets = np.argmax(targets, axis=1)

        total_num = len(predictions)
        hit_num = int(np.sum(predictions == targets))  # 正确，命中
        return {"total_num": total_num,
                "hit_num": hit_num,
                "accuracy": 1.0 * hit_num / total_num}


class MSEEvaluator(Evaluator):
    """MSEEvaluator calculates mse of predictions and targets.
    """

    @classmethod
    def evaluate(cls, predictions, targets) -> dict:
        assert predictions.shape == targets.shape

        if predictions.ndim == 1:
            mse = np.mean(np.square(predictions - targets))
        elif predictions.ndim == 2:
            mse = np.mean(np.sum(np.square(predictions - targets), axis=1))
        else:
            raise ValueError("predision supposes to have 1 or 2 dim.")

        return {"mse": mse}
