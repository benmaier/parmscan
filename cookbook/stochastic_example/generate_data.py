import numpy as np
from itertools import product

# define a few parameters
parameters = [
            ('measurement', np.arange(200)),
            ('y0', np.arange(0,10,2)),
            ('frequency', np.logspace(0,1,3)),
            ('amplitude', np.linspace(0.1,1,3)),
            ('phase', np.linspace(0,2*np.pi,5)),
            ('decay', np.linspace(0,3,4)),
        ]
pkeys, pvals = zip(*parameters)

time = np.linspace(0,1,51)

def simulation(t,
               frequency,
               amplitude,
               phase,
               decay,
               y0,
               ):
    f = frequency
    A = amplitude
    p = phase
    d = decay

    vals = A * np.exp(-d*t) * np.cos(2*np.pi*f*t+p) + y0
    vals += np.random.randn(len(vals))

    diff = np.diff(vals)/np.diff(t)
    diff = np.concatenate((
                            [np.nan],
                            diff,
                         ))
    return [vals, diff]

def get_all():

    # prepare shape of result array
    shape = [len(v) for v in pvals]
    indices = [ list(range(s)) for s in shape]

    # prepare result array (all parameters, 2 observables, one val for each time point)
    result = np.zeros(shape+[2,len(time)])

    # iterate through all combinations of indices
    for ndx in product(*indices):

        # get the parameters of this configuration
        kwargs = { key: pvals[i][ndx[i]] for i, key in enumerate(pkeys) }
        meas = kwargs.pop('measurement')

        # get corresponding simulation result
        res = simulation(time,**kwargs)

        # save values and derivative
        result[ndx] = res

    return result


if __name__ == "__main__":
    result = get_all()

    #np.save('data/result_measurements.npy',result)

    q = [2.5,25,50,75,97.5]
    k = ['2.5','25','50','75','97.5']
    percentiles = np.nanpercentile(result, q, axis=0)

    perc = { key: percentiles[i] for i,key in enumerate(k) }
    np.savez('data/result_percentiles.npz',**perc)



