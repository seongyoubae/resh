import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import copy
from scipy.stats import ttest_rel

def rollout(model, dataset, opts):
    model.eval()
    predictions = []
    with torch.no_grad():
        for data in dataset:
            pred, _ = model(data)
            predictions.append(pred)
    return torch.cat(predictions, dim=0)

def get_inner_model(model):
    return model.module if hasattr(model, "module") else model

class Baseline(object):
    def wrap_dataset(self, dataset):
        return dataset

    def unwrap_batch(self, batch):
        return batch, None

    def eval(self, x, c):
        raise NotImplementedError("Override this method")

    def get_learnable_parameters(self):
        return []

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

class WarmupBaseline(Baseline):
    def __init__(self, baseline, n_epochs=1, warmup_exp_beta=0.8):
        super(WarmupBaseline, self).__init__()
        self.baseline = baseline
        assert n_epochs > 0, "n_epochs to warmup must be positive"
        self.warmup_baseline = ExponentialBaseline(warmup_exp_beta)
        self.alpha = 0
        self.n_epochs = n_epochs

    def wrap_dataset(self, dataset):
        if self.alpha > 0:
            return self.baseline.wrap_dataset(dataset)
        return self.warmup_baseline.wrap_dataset(dataset)

    def unwrap_batch(self, batch):
        if self.alpha > 0:
            return self.baseline.unwrap_batch(batch)
        return self.warmup_baseline.unwrap_batch(batch)

    def eval(self, x, c):
        if self.alpha == 1:
            return self.baseline.eval(x, c)
        if self.alpha == 0:
            return self.warmup_baseline.eval(x, c)
        v, l = self.baseline.eval(x, c)
        vw, lw = self.warmup_baseline.eval(x, c)
        return self.alpha * v + (1 - self.alpha) * vw, self.alpha * l + (1 - self.alpha) * lw

    def epoch_callback(self, model, epoch):
        self.baseline.epoch_callback(model, epoch)
        if epoch < self.n_epochs:
            self.alpha = (epoch + 1) / float(self.n_epochs)
            print("Set warmup alpha = {}".format(self.alpha))

    def state_dict(self):
        return self.baseline.state_dict()

    def load_state_dict(self, state_dict):
        self.baseline.load_state_dict(state_dict)

class NoBaseline(Baseline):
    def eval(self, x, c):
        return 0, 0

class ExponentialBaseline(Baseline):
    def __init__(self, beta):
        super(ExponentialBaseline, self).__init__()
        self.beta = beta
        self.v = None

    def eval(self, x, c):
        if self.v is None:
            v = c.mean()
        else:
            v = self.beta * self.v + (1. - self.beta) * c.mean()
        self.v = v.detach()  # detach to prevent backpropagation
        return self.v, 0

    def state_dict(self):
        return {'v': self.v}

    def load_state_dict(self, state_dict):
        self.v = state_dict['v']

class CriticBaseline(Baseline):
    def __init__(self, critic):
        super(CriticBaseline, self).__init__()
        self.critic = critic

    def eval(self, x, c):
        v = self.critic(x)
        return v.detach(), F.mse_loss(v, c.detach())

    def get_learnable_parameters(self):
        return list(self.critic.parameters())

    def epoch_callback(self, model, epoch):
        pass

    def state_dict(self):
        return {'critic': self.critic.state_dict()}

    def load_state_dict(self, state_dict):
        critic_state_dict = state_dict.get('critic', {})
        if not isinstance(critic_state_dict, dict):
            critic_state_dict = critic_state_dict.state_dict()
        self.critic.load_state_dict({**self.critic.state_dict(), **critic_state_dict})

class RolloutBaseline(Baseline):
    def __init__(self, model, problem, opts, epoch=0):
        super(RolloutBaseline, self).__init__()
        self.problem = problem
        self.opts = opts
        self._update_model(model, epoch)

    def _update_model(self, model, epoch, dataset=None):
        self.model = copy.deepcopy(model)
        if dataset is not None:
            if len(dataset) != self.opts.val_size:
                print("Warning: not using saved baseline dataset since val_size does not match")
                dataset = None
            elif (dataset[0] if self.problem.NAME == 'tsp' else dataset[0]['loc']).size(0) != self.opts.graph_size:
                print("Warning: not using saved baseline dataset since graph_size does not match")
                dataset = None

        if dataset is None:
            self.dataset = self.problem.make_dataset(
                size=self.opts.graph_size, num_samples=self.opts.val_size, distribution=self.opts.data_distribution)
        else:
            self.dataset = dataset
        print("Evaluating baseline model on evaluation dataset")
        self.bl_vals = rollout(self.model, self.dataset, self.opts).cpu().numpy()
        self.mean = self.bl_vals.mean()
        self.epoch = epoch

    def wrap_dataset(self, dataset):
        print("Evaluating baseline on dataset...")
        return BaselineDataset(dataset, rollout(self.model, dataset, self.opts).view(-1, 1))

    def unwrap_batch(self, batch):
        return batch['data'], batch['baseline'].view(-1)

    def eval(self, x, c):
        with torch.no_grad():
            v, _ = self.model(x)
        return v, 0

    def epoch_callback(self, model, epoch):
        print("Evaluating candidate model on evaluation dataset")
        candidate_vals = rollout(model, self.dataset, self.opts).cpu().numpy()
        candidate_mean = candidate_vals.mean()
        print("Epoch {} candidate mean {}, baseline epoch {} mean {}, difference {}".format(
            epoch, candidate_mean, self.epoch, self.mean, candidate_mean - self.mean))
        if candidate_mean - self.mean < 0:
            t, p = ttest_rel(candidate_vals, self.bl_vals)
            p_val = p / 2
            assert t < 0, "T-statistic should be negative"
            print("p-value: {}".format(p_val))
            if p_val < self.opts.bl_alpha:
                print('Update baseline')
                self._update_model(model, epoch)

    def state_dict(self):
        return {
            'model': self.model,
            'dataset': self.dataset,
            'epoch': self.epoch
        }

    def load_state_dict(self, state_dict):
        load_model = copy.deepcopy(self.model)
        get_inner_model(load_model).load_state_dict(get_inner_model(state_dict['model']).state_dict())
        self._update_model(load_model, state_dict['epoch'], state_dict['dataset'])

class BaselineDataset(Dataset):
    def __init__(self, dataset=None, baseline=None):
        super(BaselineDataset, self).__init__()
        self.dataset = dataset
        self.baseline = baseline
        assert (len(self.dataset) == len(self.baseline)), "Dataset and baseline lengths must match"

    def __getitem__(self, item):
        return {
            'data': self.dataset[item],
            'baseline': self.baseline[item]
        }

    def __len__(self):
        return len(self.dataset)
