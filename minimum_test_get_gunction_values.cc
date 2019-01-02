#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_refinement.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_q.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <iostream>

int main()
{
    // FINITE ELEMENT
    using namespace dealii;
    const int dim = 2;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler(triangulation);
    FE_Q<dim> fe(1);
    Vector<double> present_solution;

    // MESH
    const unsigned int initial_global_refinement = 1;
    GridGenerator::hyper_cube(triangulation, -1, 1);
    triangulation.refine_global(initial_global_refinement);
    dof_handler.distribute_dofs(fe);
    // print dofs information: n_active_cells & n_dofs
    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    // FEVALUES
    QGauss<dim> quadrature_formula(2);
    FEValues<dim> fe_values(fe, quadrature_formula,
                            update_values | update_gradients |
                                update_quadrature_points | update_JxW_values);

    // TEST get_function_values()
    present_solution.reinit(dof_handler.n_dofs());
    for (unsigned int i = 0; i != dof_handler.n_dofs(); ++i)
    {
        present_solution[i] = i;
    }
    const unsigned int dofs_per_cell = fe.dofs_per_cell;
    const unsigned int n_q_points = quadrature_formula.size();
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    unsigned int cell_counter = 0;
    std::vector<double> local_values(n_q_points);
    for (; cell != endc; ++cell)
    {
        cell_counter += 1;
        fe_values.reinit(cell);
        fe_values.get_function_values(present_solution, local_values);
        std::cout << "local_values at " << cell_counter << " th cell:" << std::endl;
        for (auto x : local_values)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }

    // END
    std::cout << "End." << std::endl;
}
