import enum
import math
from typing import Callable, Optional

import numpy as np
import torch as th
import random

from . import path
from .integrators import ode, sde
from .utils import mean_flat, expand_dims
from .dpm_solver import NoiseScheduleFlow, model_wrapper, DPM_Solver


class ModelType(enum.Enum):
    """
    Which type of output the model predicts.
    """

    NOISE = enum.auto()  # the model predicts epsilon
    SCORE = enum.auto()  # the model predicts \nabla \log p(x)
    VELOCITY = enum.auto()  # the model predicts v(x)


class PathType(enum.Enum):
    """
    Which type of path to use.
    """

    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    """
    Which type of weighting to use.
    """

    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:
    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps, snr_type, do_shift, seq_len,
                 dynamic_time_shift: bool = False,
                 time_shift_version: str = "v1"):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }

        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

        self.snr_type = snr_type
        self.do_shift = do_shift
        self.seq_len = seq_len
        self.dynamic_time_shift = dynamic_time_shift
        self.time_shift_version = time_shift_version
    def prior_logp(self, z):
        """
        Standard multivariate normal prior
        Assume z is batched
        """
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2.0 * np.log(2 * np.pi) - th.sum(x**2) / 2.0
        return th.vmap(_fn)(z)

    def check_interval(
        self,
        train_eps,
        sample_eps,
        *,
        diffusion_form="SBDM",
        sde=False,
        reverse=False,
        eval=False,
        last_step_size=0.0,
    ):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]) and (
            self.model_type != ModelType.VELOCITY or sde
        ):  # avoid numerical issue by taking a first semi-implicit step
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size

        if reverse:
            t0, t1 = 1 - t0, 1 - t1

        return t0, t1

    def sample(self, x1, process_index, num_processes):
        """Sampling x0 & t based on shape of x1 (if needed)
        Args:
          x1 - data point; [batch, *dim]
        """
        if isinstance(x1, (list, tuple)):
            x0 = [th.randn_like(img_start) for img_start in x1]
        else:
            x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)

        if self.snr_type.startswith("uniform"):
            assert t0 == 0.0 and t1 == 1.0, "not implemented."
            if "_" in self.snr_type:
                _, t0, t1 = self.snr_type.split("_")
                t0, t1 = float(t0), float(t1)
            t = th.rand((len(x1),)) * (t1 - t0) + t0
        if self.snr_type == "stratified_uniform":
            batch_size = len(x1)
            n = batch_size * num_processes
            offsets = th.arange(process_index, n, num_processes)
            u = th.rand(size=(batch_size,))
            t = ((offsets + u) / n)
        elif self.snr_type == "lognorm":
            u = th.normal(mean=0.0, std=1.0, size=(len(x1),))
            t = 1 / (1 + th.exp(-u)) * (t1 - t0) + t0
        elif self.snr_type == "zero":
            t = th.rand((len(x1),)) 
            for _ in range(len(x1)):
                if random.random() < 1.0:
                    t[_] = 0.0
            # print(t)
        else:
            raise NotImplementedError("Not implemented snr_type %s" % self.snr_type)

        if self.do_shift:
            if self.dynamic_time_shift:
                if self.time_shift_version == "v1":
                    base_shift: float = 0.5
                    max_shift: float = 1.15
                    lin_func = self.get_lin_function(y1=base_shift, y2=max_shift)
                        
                    mu = th.tensor([lin_func((_x1.shape[-2] // 2) * (_x1.shape[-1] // 2)) for _x1 in x1], dtype=t.dtype, device=t.device).view_as(t)
                    t = self.time_shift(mu, 1.0, t)
                elif self.time_shift_version == "v2":
                    tokens = th.tensor([(_x1.shape[-2] // 2) * (_x1.shape[-1] // 2) for _x1 in x1], dtype=t.dtype, device=t.device).view_as(t)
                    t = self.time_shift_v2(tokens, t)
            else:
                if self.time_shift_version == "v1":
                    base_shift: float = 0.5
                    max_shift: float = 1.15
                    mu = self.get_lin_function(y1=base_shift, y2=max_shift)(self.seq_len)
                    t = self.time_shift(mu, 1.0, t)
                elif self.time_shift_version == "v2":
                    tokens = th.tensor([self.seq_len] * len(x1), dtype=t.dtype, device=t.device).view_as(t)
                    t = self.time_shift_v2(tokens, t)
        t = t.to(x1[0])
        return t, x0, x1

    def time_shift(self, mu: float, sigma: float, t: th.Tensor):
        # the following implementation was original for t=0: clean / t=1: noise
        # Since we adopt the reverse, the 1-t operations are needed
        t = 1 - t
        if isinstance(mu, th.Tensor):
            t = th.exp(mu) / (th.exp(mu) + (1 / t - 1) ** sigma)
        else:
            t = math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)
        t = 1 - t
        return t
    
    def time_shift_v2(self, tokens: th.Tensor, t: th.Tensor):
        # t = th.exp(mu) / (th.exp(mu) + (1 / t - 1) ** sigma)
        m = th.sqrt(tokens) / 20
        t = t / (m - m * t + t)
        return t

    def get_lin_function(
        self, x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
    ) -> Callable[[float], float]:
        m = (y2 - y1) / (x2 - x1)
        b = y1 - m * x1
        return lambda x: m * x + b

    def training_losses(
        self,
        model,
        x1,
        model_kwargs=None,
        process_index: Optional[int] = None,
        num_processes: Optional[int] = None,
        reduction: str = 'mean',
    ):
        """Loss for training the score model
        Args:
        - model: backbone model; could be score, noise, or velocity
        - x1: datapoint
        - model_kwargs: additional arguments for the model
        """

        terms = {}

        if model_kwargs is None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1, process_index, num_processes)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)

        terms = {}
        terms['t'] = t
        terms['xt'] = xt
        
        if "cond" in model_kwargs:
            conds = model_kwargs.pop("cond")
            xt = [th.cat([x, cond], dim=0) if cond is not None else x for x, cond in zip(xt, conds)]
        model_output = model(xt, t, **model_kwargs)
        B = len(x0)

        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            if isinstance(x1, (list, tuple)):
                assert len(model_output) == len(ut) == len(x1)
                for i in range(B):
                    assert (
                        model_output[i].shape == ut[i].shape == x1[i].shape
                    ), f"{model_output[i].shape} {ut[i].shape} {x1[i].shape}"
                terms["task_loss"] = th.stack(
                    [th.nn.functional.mse_loss(ut[i].float(), model_output[i].float(), reduction=reduction) for i in range(B)],
                    dim=0,
                )
            else:
                terms["task_loss"] = mean_flat(((model_output - ut) ** 2))
        else:
            raise NotImplementedError

        terms["loss"] = terms["task_loss"]
        terms["t"] = t
        return terms

    def get_drift(self):
        """member function for obtaining the drift of the probability flow ODE"""

        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output  # by change of variable

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            model_output = model(x, t, **model_kwargs)
            return model_output

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(
        self,
    ):
        """member function for obtaining score of
        x_t = alpha_t * x + sigma_t * eps"""
        if self.model_type == ModelType.NOISE:
            score_fn = (
                lambda x, t, model, **kwargs: model(x, t, **kwargs)
                / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwagrs: model(x, t, **kwagrs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: self.path_sampler.get_score_from_velocity(
                model(x, t, **kwargs), x, t
            )
        else:
            raise NotImplementedError()

        return score_fn


class Sampler:
    """Sampler class for the transport model"""

    def __init__(
        self,
        transport,
    ):
        """Constructor for a general sampler; supporting different sampling methods
        Args:
        - transport: an tranport object specify model prediction & interpolant type
        """

        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(
        self,
        *,
        diffusion_form="SBDM",
        diffusion_norm=1.0,
    ):
        def diffusion_fn(x, t):
            diffusion = self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)
            return diffusion

        sde_drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs) + diffusion_fn(x, t) * self.score(
            x, t, model, **kwargs
        )

        sde_diffusion = diffusion_fn

        return sde_drift, sde_diffusion

    def __get_last_step(
        self,
        sde_drift,
        *,
        last_step,
        last_step_size,
    ):
        """Get the last step function of the SDE solver"""

        if last_step is None:
            last_step_fn = lambda x, t, model, **model_kwargs: x
        elif last_step == "Mean":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + sde_drift(x, t, model, **model_kwargs) * last_step_size
            )
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t  # simple aliasing; the original name was too long
            sigma = self.transport.path_sampler.compute_sigma_t
            last_step_fn = lambda x, t, model, **model_kwargs: x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][
                0
            ] * self.score(x, t, model, **model_kwargs)
        elif last_step == "Euler":
            last_step_fn = (
                lambda x, t, model, **model_kwargs: x + self.drift(x, t, model, **model_kwargs) * last_step_size
            )
        else:
            raise NotImplementedError()

        return last_step_fn

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
    ):
        """returns a sampling function with given SDE settings
        Args:
        - sampling_method: type of sampler used in solving the SDE; default to be Euler-Maruyama
        - diffusion_form: function form of diffusion coefficient; default to be matching SBDM
        - diffusion_norm: function magnitude of diffusion coefficient; default to 1
        - last_step: type of the last step; default to identity
        - last_step_size: size of the last step; default to match the stride of 250 steps over [0,1]
        - num_steps: total integration step of SDE
        """

        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form,
            diffusion_norm=diffusion_norm,
        )

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            diffusion_form=diffusion_form,
            sde=True,
            eval=True,
            reverse=False,
            last_step_size=last_step_size,
        )

        _sde = sde(
            sde_drift,
            sde_diffusion,
            t0=t0,
            t1=t1,
            num_steps=num_steps,
            sampler_type=sampling_method,
        )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            xs = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)

            assert len(xs) == num_steps, "Samples does not match the number of steps"

            return xs

        return _sample
    
    def sample_dpm(
        self,
        model,
        model_kwargs=None,
    ):

        noise_schedule = NoiseScheduleFlow(schedule="discrete_flow")

        def noise_pred_fn(x, t_continuous):
            output = model(x, 1 - t_continuous, **model_kwargs)
            _, sigma_t = noise_schedule.marginal_alpha(t_continuous), noise_schedule.marginal_std(t_continuous)
            try:
                noise = x - (1 - expand_dims(sigma_t, x.dim()).to(x)) * output
            except:
                noise = x - (1 - expand_dims(sigma_t, x.dim()).to(x)) * output[0]
            return noise

        return DPM_Solver(noise_pred_fn, noise_schedule, algorithm_type="dpmsolver++").sample


    def sample_ode(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
        reverse=False,
        do_shift=False,
        time_shifting_factor=None, 
    ):
        """returns a sampling function with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        # for flux
        drift = lambda x, t, model, **kwargs: self.drift(x, t, model, **kwargs)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=reverse,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
            do_shift=do_shift,
            time_shifting_factor=time_shifting_factor,
        )

        return _ode.sample

    def sample_ode_likelihood(
        self,
        *,
        sampling_method="dopri5",
        num_steps=50,
        atol=1e-6,
        rtol=1e-3,
    ):
        """returns a sampling function for calculating likelihood with given ODE settings
        Args:
        - sampling_method: type of sampler used in solving the ODE; default to be Dopri5
        - num_steps:
            - fixed solver (Euler, Heun): the actual number of integration steps performed
            - adaptive solver (Dopri5): the number of datapoints saved during integration; produced by interpolation
        - atol: absolute error tolerance for the solver
        - rtol: relative error tolerance for the solver
        """

        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return (-drift, logp_grad)

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps,
            self.transport.sample_eps,
            sde=False,
            eval=True,
            reverse=False,
            last_step_size=0.0,
        )

        _ode = ode(
            drift=_likelihood_drift,
            t0=t0,
            t1=t1,
            sampler_type=sampling_method,
            num_steps=num_steps,
            atol=atol,
            rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            input = (x, init_logp)
            drift, delta_logp = _ode.sample(input, model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            logp = prior_logp - delta_logp
            return logp, drift

        return _sample_fn
