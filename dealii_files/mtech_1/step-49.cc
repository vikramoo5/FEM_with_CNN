/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2013 - 2021 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE.md at
 * the top level directory of deal.II.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Timo Heister, Texas A&M University, 2013
 */


// This tutorial program is odd in the sense that, unlike for most other
// steps, the introduction already provides most of the information on how to
// use the various strategies to generate meshes. Consequently, there is
// little that remains to be commented on here, and we intersperse the code
// with relatively little text. In essence, the code here simply provides a
// reference implementation of what has already been described in the
// introduction.

// @sect3{Include files}

//#include <deal.II/grid/tria.h>
//#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

//#include <iostream>
//#include <fstream>

#include <map>
#include <thread>
#include <chrono>

//using namespace dealii;



#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>
using namespace dealii;

// @sect3{Generating output for a given mesh}

// The following function generates some output for any of the meshes we will
// be generating in the remainder of this program. In particular, it generates
// the following information:
//
// - Some general information about the number of space dimensions in which
//   this mesh lives and its number of cells.
// - The number of boundary faces that use each boundary indicator, so that
//   it can be compared with what we expect.
//
// Finally, the function outputs the mesh in VTU format that can easily be
// visualized in Paraview or VisIt.


class Step3
{
public:
  Step3();
  void run();
  void run_distorted(double value);

  void data_out();
  std::vector<int> boundary_mark;
  std::vector<double> boundary_value;
  double temp_left=0;
  double temp_right=0;
  double temp_top=0;
  double temp_bottom=0;
  std::string file_mesh="mesh_folder/rec_4.msh";
  std::string file_save="new_folder/test1.txt";
private:
  void grid_1();
  //void make_grid();
  void setup_system();
  void assemble_system();
  void solve();
  void output_results() const;
  void random_distort(Triangulation<2> &triangulation,double value);
  template <int dim>
  void print_mesh_info(const Triangulation<dim> &triangulation,const std::string &  filename);
  Triangulation<2> triangulation;
  FE_Q<2>          fe;
  DoFHandler<2>    dof_handler;
  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;
  Vector<double> solution;
  Vector<double> system_rhs;

  // Some variable like k11.k12,k21,k22 and force
  double k11=1;
  double k12=0;
  double k21=0;
  double k22=1;
  double f=1000;

};




Step3::Step3()
  : fe(1)
  , dof_handler(triangulation)
{}

void Step3::grid_1()
{
  //Triangulation<2> triangulation;

  GridIn<2> gridin;
  gridin.attach_triangulation(triangulation);
  std::ifstream f(file_mesh);
  gridin.read_msh(f);

  print_mesh_info(triangulation, "grid-1.vtu");
}


void Step3::setup_system()
{
  dof_handler.distribute_dofs(fe);
  std::cout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  sparsity_pattern.copy_from(dsp);
  system_matrix.reinit(sparsity_pattern);
  solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
}


void Step3::assemble_system()
{
  QGauss<2> quadrature_formula(fe.degree+6);

  FEValues<2> fe_values(fe,
                        quadrature_formula,
                        update_values | update_gradients | update_JxW_values | update_quadrature_points);
  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double>     cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);


  int dim=2;

  std::vector<std::vector<double>> K_conductivity(2,std::vector<double>(2,0));

  double force=f;

 // int iter=0;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;


      // The following code in the commented section shoes how to code functions varying with x and y
      // x=fe_values.quadrature_point(q_index)[0]
      // y=fe_values.quadrature_point(q_index)[1]

      /*  float integral=0;

    	  for (const unsigned int q_index : fe_values.quadrature_point_indices()){

    		   std::cout<<fe_values.quadrature_point(q_index)<<std::endl;
               integral=integral+pow(fe_values.quadrature_point(q_index)[0],2)*fe_values.quadrature_point(q_index)[1]*fe_values.JxW(q_index);
         }
    	  std::cout<<"integral: "<<integral<<std::endl;
       */






      for (const unsigned int q_index : fe_values.quadrature_point_indices())
        {
    	  //K_conductivity[0][0]=10*pow(fe_values.quadrature_point(q_index)[0],6);    // x*y=fe_values.quadrature_point(q_index)[0]*fe_values.quadrature_point(q_index)[1]
    	  K_conductivity[0][0]=k11;
    	  K_conductivity[0][1]=k12;
    	  K_conductivity[1][0]=k21;
          K_conductivity[1][1]=k22;

          for (const unsigned int i : fe_values.dof_indices()){
            for (const unsigned int j : fe_values.dof_indices()){
            	for(int k=0;k<dim;k++){
            		for(int l=0;l<dim;l++){
              cell_matrix(i, j) +=
                (fe_values.shape_grad(i, q_index)[k] * K_conductivity[k][l]*// grad phi_i(x_q)
                 fe_values.shape_grad(j, q_index)[l] * // grad phi_j(x_q)
                 fe_values.JxW(q_index));           // dx
             // fe_values.quadrature_point(0)[0]

            		}
            	}


            }

          }
          for (const unsigned int i : fe_values.dof_indices())
                                            cell_rhs(i) += (fe_values.shape_value(i, q_index) * // phi_i(x_q)
                                                            force *                                // f(x_q)
                                                            fe_values.JxW(q_index));   // dx

        }
      cell->get_dof_indices(local_dof_indices);
      for (const unsigned int i : fe_values.dof_indices())
        for (const unsigned int j : fe_values.dof_indices())
          system_matrix.add(local_dof_indices[i],
                            local_dof_indices[j],
                            cell_matrix(i, j));
      for (const unsigned int i : fe_values.dof_indices())
        system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }
  std::map<types::global_dof_index, double> boundary_values;

  // Setting dirichlet value of 10 at left ad right whose boundary value number is 0

  // Setting dirichlette value of 20 at top and bottom whose boundary value number is 1
  int size_of_boundary=boundary_value.size();

  for(int i=0;i<size_of_boundary;i++){
	  VectorTools::interpolate_boundary_values(dof_handler,
	                                               boundary_mark[i],
	                                               ConstantFunction<2>(boundary_value[i]),
	                                               boundary_values);

  }

 /* VectorTools::interpolate_boundary_values(dof_handler,
                                             6,
                                             ConstantFunction<2>(temp_left),
                                             boundary_values);

  VectorTools::interpolate_boundary_values(dof_handler,
                                               7,
                                               ConstantFunction<2>(temp_top),
                                               boundary_values);

 VectorTools::interpolate_boundary_values(dof_handler,
                                               8,
                                               ConstantFunction<2>(temp_right),
                                               boundary_values);
  VectorTools::interpolate_boundary_values(dof_handler,
                                               9,
                                               ConstantFunction<2>(temp_bottom),
                                               boundary_values);



  VectorTools::interpolate_boundary_values(dof_handler,
                                           10,
                                           ConstantFunction<2>(400),
                                           boundary_values);                  */


  MatrixTools::apply_boundary_values(boundary_values,
                                     system_matrix,
                                     solution,
                                     system_rhs);
}


void Step3::solve()
{
  SolverControl solver_control(10000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());
}


void Step3::output_results() const
{
  DataOut<2> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtk");
  data_out.write_vtk(output);
  //std::ofstream file("new_folder/test.txt");
  //file<<"hello mybbjhbjhb dear vikram";
  //file.close();


}





template <int dim>
void Step3::print_mesh_info(const Triangulation<dim> &triangulation,
                     const std::string &       filename)
{
  std::cout << "Mesh info:" << std::endl
            << " dimension: " << dim << std::endl
            << " no. of cells: " << triangulation.n_active_cells() << std::endl;

  // Next loop over all faces of all cells and find how often each
  // boundary indicator is used (recall that if you access an element
  // of a std::map object that doesn't exist, it is implicitly created
  // and default initialized -- to zero, in the current case -- before
  // we then increment it):
  {
    std::map<types::boundary_id, unsigned int> boundary_count;
    for (const auto &face : triangulation.active_face_iterators())
      if (face->at_boundary())
        boundary_count[face->boundary_id()]++;

    std::cout << " boundary indicators: ";
    for (const std::pair<const types::boundary_id, unsigned int> &pair :
         boundary_count)
      {
        std::cout << pair.first << "(" << pair.second << " times) ";
      }
    std::cout << std::endl;
  }

  // Finally, produce a graphical representation of the mesh to an output
  // file:
  std::ofstream out(filename);
  GridOut       grid_out;
  grid_out.write_vtu(triangulation, out);
  std::cout << " written to " << filename << std::endl << std::endl;
}



void Step3::data_out()
{

	std::ofstream file(file_save);
	file<<"This is start of the file->"<<std::endl;
	file<<"start"<<std::endl;

	// Geting DOF_per_cell,dimension and local to global node conversion(local_dof_indices)
	const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
	std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
	const int dim=2;
	//std::cout<<"dof_per_cell="<<dofs_per_cell<<std::endl;

	//distributing degree of freedom from finite element to dof . These following four lines will help in obtaining coordinates of every points
	dof_handler.distribute_dofs (fe);
	MappingQ1<dim,dim> mapping;
	std::vector< Point<dim,double> > dof_coords(dof_handler.n_dofs());
	DoFTools::map_dofs_to_support_points<dim,dim>(mapping,dof_handler,dof_coords);




	// making something called connectivity matrix which keep informatio  of everything from coordinates to temperature
	std::map<int,std::vector<double>> data;
	std::map<int,std::set<int>> connectivity;
    int no_of_nodes=0;



	for (const auto &cell : dof_handler.active_cell_iterators()){

		cell->get_dof_indices(local_dof_indices);
		//std::cout<<local_dof_indices[0]<<local_dof_indices[1]<<local_dof_indices[2]<<local_dof_indices[3]<<std::endl;
		for ( int i=0;i<dofs_per_cell;i++){


			if (data.find(local_dof_indices[i])==data.end()) {

                no_of_nodes+=1;


				//Pushing x coordinate and y coordinate value of the node
				data[local_dof_indices[i]].push_back(dof_coords[local_dof_indices[i]][0]);
				//std::cout<<"x -- cooo "<<dof_coords[local_dof_indices[i]][0]<<std::endl;
				data[local_dof_indices[i]].push_back(dof_coords[local_dof_indices[i]][1]);

				//Writing temperature value at appropriate place in the connectivity matrix
				data[local_dof_indices[i]].push_back(solution[local_dof_indices[i]]);

                // By default assuming that node is not present on boundary we will change it later

				data[local_dof_indices[i]].push_back(0);


				// writing k11,k12,k21 and k22 and force values in the connectivity array
				data[local_dof_indices[i]].push_back(f);     //force
				data[local_dof_indices[i]].push_back(k11);     //k11
				data[local_dof_indices[i]].push_back(k12);     //k12
				data[local_dof_indices[i]].push_back(k21);     //k21
				data[local_dof_indices[i]].push_back(k22);     //k22



			}



			// getting connectivity pattern between different nodes of the mesh
			if(i==0){
			connectivity[local_dof_indices[0]].insert(local_dof_indices[1]);
			connectivity[local_dof_indices[0]].insert(local_dof_indices[2]);
			}
			else if(i==1){
			connectivity[local_dof_indices[1]].insert(local_dof_indices[0]);
			connectivity[local_dof_indices[1]].insert(local_dof_indices[3]);
			}
			else if(i==2){
			connectivity[local_dof_indices[2]].insert(local_dof_indices[0]);
			connectivity[local_dof_indices[2]].insert(local_dof_indices[3]);
			}
			else if(i==3){
			connectivity[local_dof_indices[3]].insert(local_dof_indices[1]);
			connectivity[local_dof_indices[3]].insert(local_dof_indices[2]);
			}

		}


	}


	// Now checking whether a particular node is present on boundary or not

	std::vector<int> boundary_condition(no_of_nodes,0);


	for (const auto &cell : dof_handler.active_cell_iterators()){

		cell->get_dof_indices(local_dof_indices);

		for ( int i=0;i<dofs_per_cell;i++){


			// now checking if this
			int no_of_faces=4;
			for(int j=0;j<no_of_faces;j++){


			if(cell->at_boundary(j)){

				if(j==0 and i==0){

					boundary_condition[local_dof_indices[0]]=1;

				}
				if(j==0 and i==2){
					boundary_condition[local_dof_indices[2]]=1;
				}
				if(j==1 and i==1){
					boundary_condition[local_dof_indices[1]]=1;
				}
				if(j==1 and i==3){
					boundary_condition[local_dof_indices[3]]=1;
				}
				if(j==2 and i==1){
					boundary_condition[local_dof_indices[1]]=1;
				}
				if(j==2 and i==0){
					boundary_condition[local_dof_indices[0]]=1;
				}
				if(j==3 and i==2){
					boundary_condition[local_dof_indices[2]]=1;
				}
				if(j==3 and i==3){
					boundary_condition[local_dof_indices[3]]=1;
				}


			}

			}



		}

	}


	//Updating data corresponding to boundary condition values to 1 for those nodes which were found to be on boundary of the domain

	for(int i=0;i<no_of_nodes;i++){
		data[i][3]=boundary_condition[i];
	}


     // writing everything in a file

	std::map<int,std::vector<double>>::iterator itr;

	for (itr = data.begin(); itr != data.end(); ++itr) {

	        file<<itr->first<<":"<<" ";


	        std::vector<double>::iterator itr_set;

	        for (itr_set = itr->second.begin(); itr_set != itr->second.end(); itr_set++) {
	        	file<<*itr_set<<" ";

	        }

	        for (auto& it : connectivity) {

	                if (it.first == itr->first) {
	                    std::set<int>::iterator itr_set;
	                    for(itr_set=it.second.begin();itr_set!=it.second.end();itr_set++){
	                    	file<<*itr_set<<" ";

	                    }
	                }
	            }
            file<<" ";
	        file<<std::endl;


	 }


     std::cout<<"size of connectivity=="<<data.size();



}

void Step3::random_distort(Triangulation<2> &triangulation,double value)
{


  GridTools::distort_random(value, triangulation, true);
  print_mesh_info(triangulation, "grid-7.vtu");
}




void Step3::run()
{



  grid_1();
  //make_grid();
 // random_distort(triangulation,value);
  setup_system();
  assemble_system();
  solve();
  output_results();
  data_out();


}

void Step3::run_distorted(double value)
{



  grid_1();
  //make_grid();
  random_distort(triangulation,value);
  setup_system();
  assemble_system();
  solve();
  output_results();
  data_out();


}

void p1(){
	for(int i=0;i<1;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=500-(i/3);
	  	  	  laplace_problem.temp_right=700-i;
	  	  	  laplace_problem.temp_top=200+(i);
	  	  	  laplace_problem.temp_bottom=700-(i/2);

	  	  	int mark[] = { 3,4};
		    int num = sizeof(mark) / sizeof(mark[0]);

		    int value[] = { 200, 700};
		    //int num = sizeof(value) / sizeof(value[0]);

		    for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		    }

	  	  	  std::string file_mesh_name="plan12";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="left_1/"+file_save+".txt";
	  	  	  double distortion=0;
	  	  	  laplace_problem.run_distorted(distortion);
	  	  	  //std::thread t1(laplace_problem.run_distorted,distortion);


	  	  	  }

}

void p2(){

	  for(int i=0;i<1;i++){
		  Step3 laplace_problem;

		  laplace_problem.temp_left=300+(i/2);
		  laplace_problem.temp_right=200+(i/4);
		  laplace_problem.temp_top=200+(i);
		  laplace_problem.temp_bottom=700-i;

		  int mark[] = { 3,4};
		  int num = sizeof(mark) / sizeof(mark[0]);

		  int value[] = { 400,650};
		  //int num = sizeof(value) / sizeof(value[0]);

		  for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		  }



		  std::string file_mesh_name="plan13";
		  //std::cout<<"i="<<std::to_string(i)<<std::endl;
		  std::string file_save="test"+std::to_string(i);
		  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
		  laplace_problem.file_save="normal/"+file_save+".txt";
		  //double distortion=-0.3+(i*0.006);
		  laplace_problem.run();


		  }

}

void p3(){

	 for(int i=0;i<1;i++){
		  	  	  	  Step3 laplace_problem;
		  	  	      laplace_problem.temp_left=200+(i/2);
		  	  	  	  laplace_problem.temp_right=600-(i/2);
		  	  	  	  laplace_problem.temp_top=700-i;
		  	  	  	  laplace_problem.temp_bottom=700-(i/2);

		  	  	      int mark[] = { 3,4 };
		  	  	  	  int num = sizeof(mark) / sizeof(mark[0]);

		  	  	  	  int value[] = { 300,600};
					  //int num = sizeof(value) / sizeof(value[0]);

					  for(int i=0;i<num;i++){
						  laplace_problem.boundary_mark.push_back(mark[i]);
						  laplace_problem.boundary_value.push_back(value[i]);
					  }


		  	  	  	  std::string file_mesh_name="plan14";
		  	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
		  	  	  	  std::string file_save="test"+std::to_string(i);
		  	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
		  	  	  	  laplace_problem.file_save="right_1/"+file_save+".txt";
		  	  	  	  double distortion=0;
		  	  	  	  laplace_problem.run_distorted(distortion);


		  	  	  	  }

}


void p4(){


for(int i=0;i<1;i++){
	  	  Step3 laplace_problem;
	      laplace_problem.temp_left=480;
		  laplace_problem.temp_right=600-i*4;
		  laplace_problem.temp_top=700-(i/3);
		  laplace_problem.temp_bottom=600;


		  int mark[] = {3,4 };
		  int num = sizeof(mark) / sizeof(mark[0]);

		  int value[] = {650,250};
		  //int n = sizeof(value) / sizeof(value[0]);

		  for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		  }

	  	  std::string file_mesh_name="plan15";
	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  std::string file_save="test"+std::to_string(i);
	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  laplace_problem.file_save="distorted/"+file_save+".txt";
	  	  double distortion=-0.05+(i*0.001);;
	  	  laplace_problem.run_distorted(distortion);


	  	  }


}

void p5(){
	for(int i=0;i<1;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=500-(i/3);
	  	  	  laplace_problem.temp_right=700-i;
	  	  	  laplace_problem.temp_top=200+(i);
	  	  	  laplace_problem.temp_bottom=700-(i/2);

	  	  	int mark[] = { 2 };
		    int num = sizeof(mark) / sizeof(mark[0]);

		    int value[] = {850};
		    //int num = sizeof(value) / sizeof(value[0]);

		    for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		    }

	  	  	  std::string file_mesh_name="big_circle_7000";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="left_2/"+file_save+".txt";
	  	  	  double distortion=0;
	  	  	  laplace_problem.run_distorted(distortion);
	  	  	  //std::thread t1(laplace_problem.run_distorted,distortion);


	  	  	  }

}

void p6(){
	for(int i=0;i<1;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=500-(i/3);
	  	  	  laplace_problem.temp_right=700-i;
	  	  	  laplace_problem.temp_top=200+(i);
	  	  	  laplace_problem.temp_bottom=700-(i/2);

	  	  	int mark[] = { 10,11,12,13,14 };
		    int num = sizeof(mark) / sizeof(mark[0]);

		    int value[] = { 800,190,200,500,300};
		    //int num = sizeof(value) / sizeof(value[0]);

		    for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		    }

	  	  	  std::string file_mesh_name="circle_rec_9000";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="right_2/"+file_save+".txt";
	  	  	  double distortion=0;
	  	  	  laplace_problem.run_distorted(distortion);
	  	  	  //std::thread t1(laplace_problem.run_distorted,distortion);


	  	  	  }

}

void p7(){
	for(int i=0;i<500;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=500-(i/3);
	  	  	  laplace_problem.temp_right=700-i;
	  	  	  laplace_problem.temp_top=200+(i);
	  	  	  laplace_problem.temp_bottom=700-(i/2);

	  	  	int mark[] = { 10,11,12,13,14,15,16,17,18 };
		    int num = sizeof(mark) / sizeof(mark[0]);

		    int value[] = { 200+(i/2), 300, 700-(i/2) ,640-(i/3),500,200+(i/5),420+(i/5),700-(i/3),430};
		    //int num = sizeof(value) / sizeof(value[0]);

		    for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		    }

	  	  	  std::string file_mesh_name="heart_circle_circle_5000";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="left_3/"+file_save+".txt";
	  	  	  double distortion=0;
	  	  	  laplace_problem.run_distorted(distortion);
	  	  	  //std::thread t1(laplace_problem.run_distorted,distortion);


	  	  	  }

}

void p8(){
	for(int i=0;i<1;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=500-(i/3);
	  	  	  laplace_problem.temp_right=700-i;
	  	  	  laplace_problem.temp_top=200+(i);
	  	  	  laplace_problem.temp_bottom=700-(i/2);

	  	  	int mark[] = { 3,4 };
		    int num = sizeof(mark) / sizeof(mark[0]);

		    int value[] = { 190,850};
		    //int num = sizeof(value) / sizeof(value[0]);

		    for(int i=0;i<num;i++){
			  laplace_problem.boundary_mark.push_back(mark[i]);
			  laplace_problem.boundary_value.push_back(value[i]);
		    }

	  	  	  std::string file_mesh_name="heart_ellipse_5000";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="right_3/"+file_save+".txt";
	  	  	  double distortion=0;
	  	  	  laplace_problem.run_distorted(distortion);
	  	  	  //std::thread t1(laplace_problem.run_distorted,distortion);


	  	  	  }

}





int main()
{
  try{


   // p1();

	auto start =std::chrono::high_resolution_clock::now();

    std::thread t1(p1);
    std::thread t2(p2);
    std::thread t3(p3);
    std::thread t4(p4);
    std::thread t5(p5);
    std::thread t6(p6);
    //std::thread t7(p7);
    std::thread t8(p8);

    t1.join();
    t2.join();
    t3.join();
    t4.join();
    t5.join();
    t6.join();
    //t7.join();
    t8.join();



    auto end =std::chrono::high_resolution_clock::now();
    auto duration =std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout<<"Total Duration-->"<<duration.count()/(1000000)<<std::endl;





	 /* for(int i=0;i<500;i++){
	  	  	  Step3 laplace_problem;
	  	  	  laplace_problem.temp_left=200+i;
	  	  	  laplace_problem.temp_right=1000-30*i;
	  	  	  //laplace_problem.temp_top=200+i;
	  	  	  laplace_problem.temp_bottom=1000-10*i;
	  	  	  std::string file_mesh_name="big_circle_150";
	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  laplace_problem.file_save="left_2/"+file_save+".txt";
	  	  	  double distortion=-0.20;
	  	  	  laplace_problem.run_distorted(distortion);


	  	  	  }*/

	 /* for(int i=0;i<500;i++){
	  	  	  	  	  Step3 laplace_problem;
	  	  	  	  	  laplace_problem.temp_left=200+i;
	  	  	  	  	  laplace_problem.temp_right=1000-30*i;
	  	  	  	  	  //laplace_problem.temp_top=200+i;
	  	  	  	  	  laplace_problem.temp_bottom=1000-10*i;
	  	  	  	  	  std::string file_mesh_name="big_circle_150";
	  	  	  	  	  //std::cout<<"i="<<std::to_string(i)<<std::endl;
	  	  	  	  	  std::string file_save="test"+std::to_string(i);
	  	  	  	  	  laplace_problem.file_mesh="mesh_folder/"+file_mesh_name+".msh";
	  	  	  	  	  laplace_problem.file_save="right_2/"+file_save+".txt";
	  	  	  	  	  double distortion=0.20;
	  	  	  	  	  laplace_problem.run_distorted(distortion);


	  	  	  	  	  }*/

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
}
