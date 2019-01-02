// test transfer
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
#include <deal.II/numerics/solution_transfer.h>

#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

int main()
{
    // FINITE ELEMENT
    using namespace dealii;
    const int dim = 2;
    Triangulation<dim> triangulation;
    DoFHandler<dim> dof_handler(triangulation);
    FE_Q<dim> fe(1);
    Vector<double> previous_solution;
    Vector<double> present_solution;

    // INITIAL MESH
    GridGenerator::hyper_cube(triangulation, -1, 1);
    dof_handler.distribute_dofs(fe);
    present_solution.reinit(dof_handler.n_dofs());
    for (unsigned int i = 0; i != dof_handler.n_dofs(); ++i)
    {
        present_solution[i] = i;
    }
    // print dofs information: n_active_cells & n_dofs
    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;

    
    previous_solution = present_solution;
    // OUTPUT-previous_solution
    {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (previous_solution, "previous_solution");
    data_out.build_patches ();
    std::ofstream output ("previous_solution.vtk");
    data_out.write_vtk (output);
    }


    // SET REFINE FLAG
    typename DoFHandler<dim>::active_cell_iterator cell = dof_handler.begin_active(),
                                                   endc = dof_handler.end();
    for (;cell != endc;++cell)
        cell->set_refine_flag();

    // TRANSFOR
    // void SolutionTransfer< dim, VectorType, DoFHandlerType >::interpolate(const VectorType &in, VectorType &out)
    //  Calling this function is allowed only if first Triangulation::prepare_coarsening_and_refinement, 
    //  second SolutionTransfer::prepare_for_coarsening_and_refinement, 
    //  an then third Triangulation::execute_coarsening_and_refinement are called before. Multiple calling of this function is NOT allowed. 
    //  Interpolating several functions can be performed in one step.
    SolutionTransfer<dim> solution_trans(dof_handler);
    std::cout << "1" << std::endl;
    triangulation.prepare_coarsening_and_refinement();
    std::cout << "2" << std::endl;
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
    std::cout << "3" << std::endl;
    triangulation.execute_coarsening_and_refinement();
    std::cout << "4" << std::endl;
    dof_handler.distribute_dofs(fe); 
    std::cout << "5" << std::endl;
    present_solution.reinit(dof_handler.n_dofs());
    std::cout << "6" << std::endl;
    solution_trans.interpolate(previous_solution, present_solution); 
    std::cout << "7" << std::endl;
     
    // print dofs information: n_active_cells & n_dofs
    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;


    // OUTPUT VTK    
    // OUTPUT-previous_solution
    std::cout << "previous_solution:" << std::endl;
    for (auto x : previous_solution)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
    std::cout << "-------------------------------------------" << std::endl;
    std::cout << "present_solution:" << std::endl;
    for (auto x : present_solution)
    {
        std::cout << x << " ";
    }

    // OUTPUT-present_solution
    {
    DataOut<dim> data_out;
    data_out.attach_dof_handler (dof_handler);
    data_out.add_data_vector (present_solution, "present_solution");
    data_out.build_patches ();
    std::ofstream output ("present_solution.vtk");
    data_out.write_vtk (output);
    }
    // END
    std::cout << "End." << std::endl;
}
