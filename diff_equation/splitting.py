# Klein Gordon equation: u_tt=alpha*u_xx -beta(x)*u, alpha>0, beta(x)>0
# TODO: use strang splitting to solve klein gordon equation

dt = 0.1  # time step size


# Strang splitting:
# from starting values u(t0,x)=g(x), u_t(t0,x)=h(x)
# solve wave equation to v(t0+dt/2,x), calculate v_t(t0+dt/2,x)
# with these as starting values solve linear hyperbolic ode with mol to w(t0+dt,x), calculate w_t(t0+dt,x)
# using these as starting values finally solve wave equation again to u(t0+dt,x)

#TODO how to get v_t and w_t? We want second order!? central differences (so also calculate v(t0+dt,x),...)?
#TODO instead of solving wave twice faster and equally accurate to solve linhyp twice?