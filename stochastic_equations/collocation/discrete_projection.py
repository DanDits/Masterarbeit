from diff_equation.splitting import make_klein_gordon_leapfrog_splitting
from polynomial_chaos.poly_chaos_distributions import get_chaos_by_distribution
from stochastic_equations.collocation.util import check_distribution_assertions

# expectancy accuracy does not depend on max_poly_degree, only on the quadrature nodes count
# the variance error decreases for higher poly degrees, as long as the quadrature nodes count is higher (high enough)
def discrete_projection_expectancy(trial, max_poly_degree, random_space_quadrature_nodes_count, spatial_domain, grid_size,
                                start_time, stop_time, delta_time):
    distr = trial.variable_distributions[0]
    chaos = get_chaos_by_distribution(distr)
    check_distribution_assertions(distr)
    basis = [chaos.poly_basis(degree) for degree in range(max_poly_degree + 1)]

    quad_nodes, quad_weights = chaos.nodes_and_weights(random_space_quadrature_nodes_count)

    poly_weights = []
    splitting_xs = None
    splitting_xs_mesh = None

    for i, poly in enumerate(basis):
        curr_poly_weight = 0
        for node, weight in zip(quad_nodes, quad_weights):
            trial.set_random_values([node])
            splitting = make_klein_gordon_leapfrog_splitting(spatial_domain, [grid_size], start_time,
                                                             trial.start_position,
                                                             trial.start_velocity, trial.alpha, trial.beta)
            splitting.progress(stop_time, delta_time, 0)
            if splitting_xs is None:
                splitting_xs = splitting.get_xs()
                splitting_xs_mesh = splitting.get_xs_mesh()
            curr_poly_weight += poly(node) * weight * splitting.solutions()[-1].real
        poly_weights.append(curr_poly_weight / chaos.normalization_gamma(i))
    expectancy = chaos.normalization_gamma(0) * poly_weights[0]
    variance = (sum((weight ** 2) * chaos.normalization_gamma(i) for i, weight in enumerate(poly_weights))
                - (chaos.normalization_gamma(0) * poly_weights[0]) ** 2)

    def poly_approximation(y):
        return sum(w * p(y) for w, p in zip(poly_weights, basis))

    return splitting_xs, splitting_xs_mesh, expectancy, variance
