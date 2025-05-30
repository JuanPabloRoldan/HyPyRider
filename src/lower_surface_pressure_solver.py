import os
import numpy as np
from stl import mesh
import pyvista as pv
from scipy.interpolate import interp1d
from velocity_altitude_map import calculate_pressure, calculate_dynamic_pressure

class SurfaceMeshAnalyzer:
    def __init__(self, file_path):
        """
        """
        self.mesh = mesh.Mesh.from_file(file_path)
        self.cell_areas = None
        self.normal_vectors = None
        self.angles = None

    def calculate_cell_area(self):
        """
        Calculate the area of each cell in the lower surface mesh.
        """
        # # Calculate the area of each cell in the lower surface mesh
        # cell_areas = numpy.linalg.norm(numpy.cross(lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 1], lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 2])) / 2
        # return cell_areas
        V0 = self.mesh.vectors[:, 0]  # First vertex of each triangle
        V1 = self.mesh.vectors[:, 1]  # Second vertex
        V2 = self.mesh.vectors[:, 2]  # Third vertex

        # Compute cross product of two edge vectors
        cross_product = np.cross(V1 - V0, V2 - V0)

        # Compute triangle area (magnitude of cross-product divided by 2)
        self.cell_areas = 0.5 * np.linalg.norm(cross_product, axis=1)

    def calculate_normal_vector(self):
        """
        Calculate the normal vector of each cell in the lower surface mesh,
        ensuring the y-component is always negative and normalizing the vectors.
        """
        V0 = self.mesh.vectors[:, 0]
        V1 = self.mesh.vectors[:, 1]
        V2 = self.mesh.vectors[:, 2]

        # Compute normal vectors using the right-hand rule
        normal_vectors = np.cross(V1 - V0, V2 - V0)

        # Identify normals with a positive y-component
        flip_mask = normal_vectors[:, 1] > 0

        # Negate incorrect normals
        normal_vectors[flip_mask] *= -1

        # Normalize the vectors (avoid division by zero)
        norms = np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]
        norms[norms == 0] = 1  # Prevent division by zero
        self.normal_vectors = normal_vectors / norms

    def calculate_angle_from_normal_vector(self, freestream_direction):
        """
        Calculate the angle from the normal vector to the free-stream direction.
        """
        # # Calculate the angle from the normal vector to the free-stream direction
        # angle_from_normal_vector = numpy.arccos(numpy.dot(normal_vectors, [0, 0, 1]) / (numpy.linalg.norm(normal_vectors) * numpy.linalg.norm([0, 0, 1])))
        # return angle_from_normal_vector

        if self.normal_vectors is None:
            self.calculate_normal_vector()

        freestream_direction = np.array(freestream_direction)  # Ensure it's a NumPy array
        freestream_direction = freestream_direction.astype(np.float64)  # Convert to float
        freestream_direction /= np.linalg.norm(freestream_direction)  # Normalize

        # Compute dot product row-wise
        freestream_direction *= -1  # Flip direction
        dot_products = np.einsum('ij,j->i', self.normal_vectors, freestream_direction)

        # Compute angles using arccos
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
        # angles = np.radians(180) - angles
        angles = np.radians(90) - angles

        return angles

    def analyze_mesh(self, freestream_direction):
        """
        Computes cell areas, normal vectors, and angles for a given freestream direction.
        """
        self.calculate_cell_area()
        self.calculate_normal_vector()
        self.angles = self.calculate_angle_from_normal_vector(freestream_direction)

    def calculate_exact_pressure_coefficients_cell(self, TM_tabulation, stag_properties, p_inf, q_inf):
        if self.angles is None:
            raise ValueError("Angles have not been computed. Run analyze_mesh() first.")

        theta_ref = TM_tabulation["Theta (radians)"].values
        mach_ref = TM_tabulation["Mach"].values
        P_P0_ref = TM_tabulation["P/P0"].values
        T_T0_ref = TM_tabulation["T/T0"].values
        rho_rho0_ref = TM_tabulation["rho/rho0"].values

        mach_interp = interp1d(theta_ref, mach_ref, kind="linear", bounds_error=False, fill_value="extrapolate")
        P_P0_interp = interp1d(theta_ref, P_P0_ref, kind="linear", bounds_error=False, fill_value="extrapolate")
        T_T0_interp = interp1d(theta_ref, T_T0_ref, kind="linear", bounds_error=False, fill_value="extrapolate")
        rho_rho0_interp = interp1d(theta_ref, rho_rho0_ref, kind="linear", bounds_error=False, fill_value="extrapolate")

        self.cell_mach = mach_interp(self.angles)
        cell_P_P0 = P_P0_interp(self.angles)
        cell_T_T0 = T_T0_interp(self.angles)
        cell_rho_rho0 = rho_rho0_interp(self.angles)

        # Look into stag_properties to get stagnation pressure (P0)
        P0 = stag_properties["P0"]

        # Find pressure per cell
        cell_P_exact = P0 * cell_P_P0
        self.cell_Cp_exact = (cell_P_exact-p_inf)/(q_inf)
        self.cell_Cl_exact = self.cell_Cp_exact * np.sin(np.radians(90) - self.angles)
        self.cell_Cd_exact = self.cell_Cp_exact * np.cos(np.radians(90) - self.angles)

    
    def calculate_newtonian_pressure_coefficients_cell(self):
        """
        Use modified newtonian theory to calculate the pressure distribution given the angle of a unit normal

        Parameters
        ----------
        Returns
            None.
        """
        
        self.cell_Cp_newtonian = 2 * np.sin(self.angles)**2
        self.cell_Cd_newtonian = 2 * np.sin(self.angles)**3
        self.cell_Cl_newtonian =  np.cos(self.angles) * 2 * np.sin(self.angles)**2

    def Cp_entire_vehicle(self, Cp_input):
        """
        Calculate normalized pressure coefficients over the entire lower surface.
        Area-weighted average of coefficients of pressure (eg: (Cp), lift (Cl), and drag (Cd)).

        Parameters
        ----------
        Cp_input: Array
            Prefered Cp distribution

        Returns
        -------
            - Cp_vehicle: Singular Cp value across entire vehicle. 
        """
        Cp_vehicle = np.sum(Cp_input*self.cell_areas)/np.sum(self.cell_areas)

        return Cp_vehicle

    def export_to_vtk(self, output_filename="outputs/output.vtk"):
        """Exports the STL surface mesh with interpolated results as a VTK file."""
        output_dir = os.path.dirname(output_filename)
        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

        # if self.cell_P_P0 is None or self.cell_T_T0 is None or self.cell_rho_rho0 is None:
        #     raise ValueError("Run calculate_exact_pressure_coefficient() first.")

        # Convert STL to PyVista Mesh
        points = self.mesh.vectors.reshape(-1, 3)  # Extract unique points
        faces = np.hstack([np.full((len(self.mesh.vectors), 1), 3), 
                           np.arange(len(points)).reshape(-1, 3)]).astype(np.int64)

        pv_mesh = pv.PolyData(points, faces)

        # Assign cell data (per triangle)
        pv_mesh.cell_data["angles"] = np.degrees(self.angles)
        pv_mesh.cell_data["Mach"] = self.cell_mach
        # pv_mesh.cell_data["P/P0"] = self.cell_P_P0
        # pv_mesh.cell_data["T/T0"] = self.cell_T_T0
        # pv_mesh.cell_data["rho/rho0"] = self.cell_rho_rho0
        # pv_mesh.cell_data["P Exact"] = self.cell_P_exact
        pv_mesh.cell_data["Cp Exact"] = self.cell_Cp_exact
        pv_mesh.cell_data["Cl Exact"] = self.cell_Cl_exact
        pv_mesh.cell_data["Cd Exact"] = self.cell_Cd_exact
        pv_mesh.cell_data["Cp Newtonian"] = self.cell_Cp_newtonian
        pv_mesh.cell_data["Cl Newtonian"] = self.cell_Cl_newtonian
        pv_mesh.cell_data["Cd Newtonian"] = self.cell_Cd_newtonian

        # For visualization of cell vector normals
        # Initialize normal vector array (set all to zero)
        sampled_normals = np.full_like(self.normal_vectors, np.nan)
        cell_ids_to_visualize = [1, 25, 50, 75, 100]
        if cell_ids_to_visualize is not None:
            for cell_id in cell_ids_to_visualize:
                if 0 <= cell_id < len(self.normal_vectors):  # Ensure ID is within range
                    sampled_normals[cell_id] = self.normal_vectors[cell_id]

        pv_mesh.cell_data["sampled_normals"] = sampled_normals  # Add to VTK

        # Save as VTK
        pv_mesh.save(output_filename)
        print(f"VTK file saved: {output_filename}")

    # def lower_surface_solver(self):
    #     """
    #     Loop through the cells in the lower surface mesh and calculate the cell area, normal vector, and angle from the normal vector to the free-stream direction.
    #     """

    #     lower_surface_mesh = import_lower_surface_mesh()
    #     cell_areas = calculate_cell_area()
    #     normal_vectors = calculate_normal_vector()
    #     angle_from_normal_vector = calculate_angle_from_normal_vector()

    #     for i in range(len(lower_surface_mesh.vectors)):
    #         print(f'Cell {i}: Area = {cell_areas[i]}, Normal Vector = {normal_vectors[i]}, Angle from Normal Vector = {angle_from_normal_vector[i]}')
   
    # def calculate_cp_modified_newtonian(self, M1):
    #     """
    #     Use modified newtonian theory to calculate the pressure distribution given the angle of a unit normal

    #     Parameters
    #     ----------
    #     M1 : float
    #         Mach number

    #     Returns
    #     -------
    #         - Cp: Pressure distribution given unit normal angle with the freestream
    #         - post_shock_stagnation_Cp: Cp downstream of the shock
    #     """
    #     p_ratio = (1 + (self.gamma - 1) / 2 * M1**2)**(-self.gamma / (self.gamma - 1))

    #     #Newtonian modified theory
    #     Cpt = (((p_ratio)*(1+((self.gamma-1)/2)*M1**2)**(self.gamma/(self.gamma-1)))-1)/(0.5*self.gamma*M1**2)
    #     Cp_newtonian_mod = Cpt*np.cos(self.angles)**2

    #     return {
    #         "Cp_newtonian_mod": Cp_newtonian_mod, #Use this one for surface calculations
    #         "post_shock_stagnation_Cp": Cpt #may be needed in future for forces
    #     }

    # def lift_over_drag(self):
    #     """
    #     Use basic newtonian theory to calculate the lift over drag value per cell or per body

    #     Parameters
    #     ----------

    #     Returns
    #     -------
    #         - Cl_Cd: Lift over Drag of the vehicle
    #     """
    #     self.Cl_Cd = self.Cl/self.Cd

    #     return{
    #         "Cl_Cd": self.Cl_Cd
    #     }
        
# Example usage:
if __name__ == "__main__":

    freestream_dir = [1, 0, 0]  # Flow along the x-axis

    analyzer = SurfaceMeshAnalyzer("src/inputs/LowerSurfaceM10-20deg-Meshed.stl")
    analyzer.analyze_mesh(freestream_dir)
    for i in range(len(analyzer.mesh.vectors)):
            print(f'Cell {i}: Area = {analyzer.cell_areas[i]:.6f}, '
                  f'Normal Vector = {analyzer.normal_vectors[i]}, '
                  f'Angle (deg) = {np.degrees(analyzer.angles[i]):.2f}')
    
    from taylor_maccoll_solver import TaylorMaccollSolver
    tm_solver = TaylorMaccollSolver(gamma=1.2)

    # Conical Shock Conditions at M=10, gamma=1.2
    theta_s = np.radians(20)  # Example shock angle
    theta_c = np.radians(18.1951829)
    Mc = 6.41062720
    V_0, Vr0, dVr0 = tm_solver.calculate_velocity_components(Mc, theta_c, theta_c)
    
    results_df = tm_solver.tabulate_from_shock_to_cone(theta_s, theta_c, Vr0, dVr0)

    altitude = 30000 # meters
    p_inf = calculate_pressure(altitude)
    q_inf = calculate_dynamic_pressure(gamma=1.2, p=p_inf, M=10)

    from oblique_shock_solver import ObliqueShockSolver
    os_solver = ObliqueShockSolver(gamma=1.2)
    os_post_shock_conditions = os_solver.calculate_post_shock_conditions(M1=10, theta_s=theta_s)
    os_M2 = os_post_shock_conditions["M2"]
    os_P2_P1 = os_post_shock_conditions["P2_P1"]
    os_P2 = os_P2_P1 * p_inf

    from isentropic_relations_solver import IsentropicRelationsSolver
    isen_solver = IsentropicRelationsSolver(gamma=1.2)
    isen_relations = isen_solver.isentropic_relations(Mach=os_M2)
    P2_P0 = isen_relations["Static Pressure Ratio (p/p0)"]
    P0_2 = os_P2 / P2_P0

    post_shock_stag_properties = {"P0": P0_2, "T0":"_", "rho0":"_"}

    analyzer.calculate_exact_pressure_coefficients_cell(results_df, post_shock_stag_properties, p_inf, q_inf)
    analyzer.calculate_newtonian_pressure_coefficients_cell()
    analyzer.export_to_vtk("src/outputs/surface_analysis.vtk")

    Cp_inputs = {
    "Cp_exact": analyzer.cell_Cp_exact,
    "Cl_exact": analyzer.cell_Cl_exact,
    "Cd_exact": analyzer.cell_Cd_exact,
    "Cp_newtonian": analyzer.cell_Cp_newtonian,
    "Cl_newtonian": analyzer.cell_Cl_newtonian,
    "Cd_newtonian": analyzer.cell_Cd_newtonian,
    }

    for name, Cp_input in Cp_inputs.items():
        total_Cp = analyzer.Cp_entire_vehicle(Cp_input=Cp_input)
        print(f"{name}: {total_Cp}")

