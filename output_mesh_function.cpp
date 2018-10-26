void output_mesh(const Triangulation<2> &triangulation, double time, unsigned int step)
{
	// std::string eps_name = "grid-";
	// std::string str1 = std::to_string(time);
	// std::string str_link = "_";
	// std::string str2 = std::to_string(step);
	// std::string eps_format = ".eps";
	std::string eps_name = "grid-"
		+ std::to_string(time)
		+ "_"
		+ std::to_string(step)
		+ ".eps";
	std::ofstream out(eps_name);
	GridOut grid_out;
	grid_out.write_eps(triangulation, out);
	std::cout << "Grid written to " << eps_name << std::endl;
}
