"""
    _check_terminal_safe(par, bpar, a_grid, a_init)

Guard against the terminal-age CRRA knife-edge: if the oldest agent receives
no labor income (`y[J] = 0`), no inheritance (`F[J] = 0`), and the asset grid
reaches zero, the EGM terminal allocation drives `c_J → 0` and `u'(c_J) → ∞`.

The positive initial `inheritances_z` guess in `solve_lifecycle_with_bequests`
only rescues this case when `F[J] > 0`; otherwise the knife-edge persists.
"""
function _check_terminal_safe(par, bpar, a_grid, a_init)
    J = par.J
    y_J = par.y[j = At(J)]
    F_J = Float64(bpar.F[end])
    amin = minimum(a_grid)
    has_terminal_income = y_J > 0
    has_terminal_inh    = F_J > 0
    safe_grid           = amin > 0
    if !(has_terminal_income || has_terminal_inh || safe_grid)
        throw(ArgumentError(
            "Terminal CRRA knife-edge: with y[J] = 0, F[J] = 0, and amin = 0, " *
            "the EGM solver will hit u'(c_J) = Inf at the lowest grid point. " *
            "Set par.y[J] > 0 (preferred — small retirement transfer), or " *
            "F[J] > 0, or amin > 0."))
    end
    return nothing
end

"""
    solve_worker_with_bequests(solver, par, bpar, a_grid; type_grid, r, w, ...)

Back-compat wrapper over `solve_lifecycle_with_bequests` with `epar = nothing`.
Solves the pure worker lifecycle with cross-type inheritance transmission.
"""
solve_worker_with_bequests(
    solver::SolutionMethod, par, bpar::BequestParams, a_grid;
    type_grid::AbstractVector{PermanentType}, kwargs...,
) = solve_lifecycle_with_bequests(solver, par, bpar, a_grid;
        type_grid, epar = nothing, kwargs...)
