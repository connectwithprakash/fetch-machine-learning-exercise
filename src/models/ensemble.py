import torch


class Ensemble:
    def __init__(self, lr_model, nn_model, window_size):
        self.window_size = window_size
        self.lr_model = lr_model
        self.nn_model = nn_model
        self.nn_model.eval()

    def predict(self, X):
        time = X[:, 0] + self.window_size
        lr_pred = self.lr_model.predict(time)
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X).float()
            nn_pred = self.nn_model(X.unsqueeze(0))
            nn_pred = nn_pred.detach().cpu().numpy()[0]

        pred = (lr_pred + nn_pred)
        return pred
        

    def __repr__(self):
        return f'Ensemble(models={self.models})'
