from diff_equation.pseudospectral_solver import WaveSolverConfig, KleinGordonMomentConfig, VelocityConfig, \
    OffsetWaveSolverConfig
from diff_equation.ode_solver import LinhypSolverConfig
import numpy as np

# Klein Gordon equation: u_tt=alpha()*u_xx -beta(x)*u, alpha()>0, beta(x)>0


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)


def make_klein_gordon_linhyp_waveoffset_configs(intervals, grid_points_list, alpha, beta, offset):
    return (LinhypSolverConfig(intervals, grid_points_list, lambda *params: (beta(*params) - offset), 0.),
            OffsetWaveSolverConfig(intervals, grid_points_list, np.sqrt(alpha()), offset))


# more or less wave_linhyp configs for wave_weight=0., except that the moment uses both moments alpha*u_xx-beta*u
# which only requires one fft/ifft pair and is therefore faster, but also unstable for bigger time step sizes.
def make_klein_gordon_leapfrog_configs(intervals, grid_points_list, alpha, beta):
    return (KleinGordonMomentConfig(intervals, grid_points_list, alpha, beta),
            VelocityConfig(intervals, grid_points_list))


def make_klein_gordon_wave_linhyp_configs(intervals, grid_points_list, alpha, beta, wave_weight):
    return (WaveSolverConfig(intervals, grid_points_list, np.sqrt(alpha()), splitting_factor=wave_weight),
            LinhypSolverConfig(intervals, grid_points_list, beta, splitting_factor=1. - wave_weight))
