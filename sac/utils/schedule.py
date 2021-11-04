### Implement Weight Decay Scheduling for Exploration as well as network training

class Schedule(object):
    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

class ConstantSchedule(object):
    def __init__(self, v):
        self.v = v
    def value(self, t):
        return v

class PiecewiseConstantSchedule(object):
    def __init__(self, endpoints):
        self._endpoints = endpoints
    def value(self, t):
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                return l
        return self._endpoints[-1][-1]

class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        outside_value: 
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

### Exponentially decaying schedule
class ExponentialSchedule(object):
    def __init__(self,initial_p=1.0, final_p=0.01,decay=0.995):
        self.decay = decay
        self.initial_p = initial_p
        self.final_p = final_p
        
    def value(self,t):
        return max(self.final_p, (self.decay**t)*self.initial_p)


def optimizer_schedule(num_timesteps):
    lr_schedule = PiecewiseSchedule(
        [
            (0, 1e-1),
            (num_timesteps / 40, 1e-1),
            (num_timesteps / 8, 5e-2),
        ],
        outside_value=5e-2,
    )
    return lr_schedule