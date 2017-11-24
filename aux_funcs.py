import numpy as np
from scipy import optimize
from matplotlib import pyplot
import kwant

def edges(sys, lat):
    """
    returns a list of sites of all the edges in sys which is defined over lat
    i.e. sites that have fewer than 4 neighbors and are not interfaced to any lead
    note: currently only works with a rectangular lattice (4 neighbors).
    """
    edge_sites = [];
    for stv in sys.site_value_pairs():
        in_interface = False
        # first check if site is an interface of any lead
        site = stv[0];
        for lead in sys.leads:
            if site in lead.interface:
                in_interface = True;
        # otherwise, check if not in bulk
        if sys.degree(site) < 4 and not in_interface:
            edge_sites.append(site)
    return edge_sites


def FD(E, mu, T):
    """
    Fermi dirac distribution at energy E with chemical potential mu and temperature T
    Assumes T, mu and E have the same units.
    """
    return 1/(1+np.exp((E-mu)/T))

def create_energy_window(mu1, mu2, T1, T2, Ngrid, cutoff):
    """
    creates a grid of energy values for calcualting finite bias quantities
    (internal function, not to be used directly unless you know what you're doing)
    """
    def FD_diff(E):
        return FD(E,mu1,T1) - FD(E,mu2,T2)

    
    # create energy window
    # find starting energy
    
    xx = np.linspace(-3,3)
    fopt = lambda E: FD_diff(E) - cutoff
    
    sol = optimize.root(fopt, min(mu1,mu2) - max(T1,T2))
    if not sol.success:
        print(sol)
    else:
        Estart = sol.x
    
    # find ending energy
    sol = optimize.root(fopt, max(mu1,mu2) + max(T1,T2))
    if not sol.success:
        print(sol)
    else:
        Estop = sol.x
    
    energies = np.linspace(Estart, Estop, Ngrid)
    return energies


def T_finite_bias(sys, params, mu1, mu2, T1, T2, lead_in, lead_out, cutoff = 1e-2, Ngrid = 10, verbose = False):
    """
    Calculate transmission between two contacts at finite bias and temperatures
    
    sys - instance of finiteSystem
    params - parameters for Hamiltonian
    mu1, mu2, T1, T2 - chemical potential and temperature of the incoming and outgoing leads
    lead_in, lead_out - incoming and outgoing leads
    cutoff - if difference of fermi functions of the 2 leads is smaller than this, current is taken to be zero
    Ngrid - number of points within non-zero window to calculate current
    
    Returns:
    the integrated transmission for the given distribution 
    """

    energies = create_energy_window(mu1, mu2, T1, T2, Ngrid, cutoff)
    
    if verbose:
        df = list(map(lambda E: FD(E,mu1,T1) - FD(E,mu2,T2), energies))
        pyplot.plot(energies, df, '-+')
        pyplot.xlabel('energies'); pyplot.ylabel('f_FD1(e) - f_FD2(e)')
        pyplot.title('FD distribution difference window')
    
    trans = []
    for energy in energies:
        smatrix = kwant.smatrix(sys, energy, args=[params])
        trans.append(smatrix.transmission(lead_out, lead_in))
    trans = np.array(trans)
    
    if verbose:
        pyplot.plot(energies, trans, '-+')
        pyplot.xlabel('energies'); pyplot.ylabel('T(e)')
        pyplot.title('Transmission')
        
    return (energies[1]-energies[0])*np.sum(trans*energies)


def wavefunc_finite_bias(sys, params, mu1, mu2, T1, T2, lead_in, cutoff = 1e-2, Ngrid = 10, verbose = False):
    """
    Calculate wavefunction for transimssion between two contacts at finite bias and temperatures
    
    sys - instance of finiteSystem
    params - parameters for Hamiltonian
    mu1, mu2, T1, T2 - chemical potential and temperature of the incoming and outgoing leads
    lead_in - incoming lead for current
    cutoff - if difference of fermi functions of the 2 leads is smaller than this, current is taken to be zero
    Ngrid - number of points within non-zero window to calculate current
    
    Returns:
    a 1-D array giving the square sum of the all the wavefunctions of all modes with the distribution difference weight 
    """
    energies = create_energy_window(mu1, mu2, T1, T2, Ngrid, cutoff)
    
    if verbose:
        df = list(map(lambda E: FD(E,mu1,T1) - FD(E,mu2,T2), energies))
        pyplot.plot(energies, df, '-+')
        pyplot.xlabel('energies'); pyplot.ylabel('f_FD1(e) - f_FD2(e)')
        pyplot.title('FD distribution difference window')
    
    density = []
    
    for energy in energies:
        wfs = kwant.wave_function(sys, energy, args=[params])
        density.append(np.sum(abs(wfs(lead_in))**2, axis=0))
        if verbose:
            print('in energy = '+str(energy), end='\r', flush=True)

    density = np.array(density)
    return np.sum(density, axis=0)


def linecut_at_dim(sys, lat, wf, dim, num_translations):
    """
    get a linecut of a wavefunction along a dimension
    
    currently only works for 2D monatomic, single component lattice
    
    sys - finalized system
    lat - lattice on which system is defined
    wf - an array representing a wavefunction with a value for each lattice site
    dim - the dimension along which to take the linecut (0 is x, 1 is y)
    num_translations - number of translations along 1-dim to move to take the line cut
    
    Returns:
    a dict, with entries:
    'positions' for the lattice sites along the cut
    'wavefunc' for the value of the wavefunction along the cut
    """
    positions = [site.pos for site in sys.sites]
    rv = lat.reduced_vecs
    pos_shift = [pos - num_translations*rv[1-dim] for pos in positions]
    
    ort_vec = np.array([rv[dim][1], -rv[dim][0]])
    
    # find linear indices of linecut
    cut_vecs_shift = list(filter(lambda x : np.dot(x,ort_vec)==0, pos_shift))
    indices = [pos_shift.index(pos) for pos in cut_vecs_shift]
    
    # extract position, wavefunction at indices
    wf_cut = np.array([wf[ind] for ind in indices])
    pos_cut = np.array([positions[ind][dim] for ind in indices])
    
    return {'positions': pos_cut, 'wavefunc' :wf_cut}