from abc import ABC, abstractmethod


class Model(ABC):
    """
    Base class for all ML models.
    """

    @abstractmethod
    def is_init(self):
        """
        Returns boolean reflecting if the model was initialised.
        :return Whether model is initialised as <bool>
        """
        pass

    @abstractmethod
    def fit(self, phi, y):
        """
        Trains the model on data.
        :param phi: features or input as (n, m) <tensor>
        :param y: targets as (n, d_out) <tensor>
        """
        pass

    @abstractmethod
    def predict_mean(self, phi):
        """
        Predicts mean at given data points.
        :param phi: features or input as (n, m) <tensor>
        :return prediction mean as (n, d_out) <tensor>
        """
        pass

    @abstractmethod
    def predict(self, phi):
        """
        Predicts mean and variance at given data points.
        :param phi: features or input as (n, m) <tensor>
        :return prediction mean and variance as (n, d_out) and (n, 1) <tensor>
        """
        pass

    @abstractmethod
    def optimise(self, max_evals=200):
        """
        Triggers hyperparameter optimisation procedure.
        :param max_evals: maximum number of optimisation iterations as <int>
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets model and gets rid of all data.
        """
        pass

    @property
    @abstractmethod
    def model_params(self):
        """
        Returns model parameters in a vector.
        :return model parameters as <tensor>
        """
        pass

    @abstractmethod
    def update_targets(self, x, y):
        """
        Updates the models by changing targets of previously seen data.
        :param x: previously seen features or inputs as (n, m) <tensor>
        :param y: new targets as (n, d_out) <tensor>
        """
        pass


class StreamingModel(Model, ABC):
    """
    Base class for all ML models that handle streaming.
    """

    @abstractmethod
    def update(self, phi, y):
        """
        Adds data to the model.
        :param phi: features or input as (n, m) <tensor>
        :param y: targets as (n, d_out) <tensor>
        """
        pass


class FeatureWrap(StreamingModel):
    """
    Wraps any ML model by automatically applying a feature map to it.
    """

    def __init__(self, model, features):
        self.model = model
        self.features = features

    def is_init(self):
        return self.model.is_init()

    def fit(self, x, y):
        phi = self.features.to_features(x)
        return self.model.fit(phi, y)

    def update(self, x, y):
        phi = self.features.to_features(x)
        return self.model.update(phi, y)

    def predict(self, x):
        phi = self.features.to_features(x)
        return self.model.predict(phi)

    def predict_mean(self, x):
        phi = self.features.to_features(x)
        return self.model.predict_mean(phi)

    def optimise(self, **kwargs):
        return self.model.optimise(**kwargs)

    def update_targets(self, x, y):
        phi = self.features.to_features(x)
        return self.model.update_targets(phi, y)

    def reset(self):
        return self.model.reset()

    @property
    def model_params(self):
        return self.model.model_params


class Features(ABC):
    @abstractmethod
    def to_features(self, x):
        """
        Transforms x to features
        :param x: input as (Tensor)
        :return: features as (Tensor)
        """
        pass
