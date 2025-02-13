import numpy as np
from stl import mesh

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
        Calculate the normal vector of each cell in the lower surface mesh.
        """
        # # Calculate the normal vector of each cell in the lower surface mesh
        # normal_vectors = numpy.cross(lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 1], lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 2])
        # return normal_vectors
        V0 = self.mesh.vectors[:, 0]
        V1 = self.mesh.vectors[:, 1]
        V2 = self.mesh.vectors[:, 2]

        # Compute normal vectors using the right-hand rule
        normal_vectors = np.cross(V1 - V0, V2 - V0)

        # Normalize the vectors (avoid division by zero)
        self.normal_vectors = normal_vectors / np.linalg.norm(normal_vectors, axis=1)[:, np.newaxis]

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
        dot_products = np.einsum('ij,j->i', self.normal_vectors, freestream_direction)

        # Compute angles using arccos (clip values to avoid numerical issues)
        angles = np.arccos(np.clip(dot_products, -1.0, 1.0))

        return angles

    def analyze_mesh(self, freestream_direction):
        """
        Computes cell areas, normal vectors, and angles for a given freestream direction.
        """
        self.calculate_cell_area()
        self.calculate_normal_vector()
        self.angles = self.calculate_angle_from_normal_vector(freestream_direction)

    def lower_surface_solver():
        """
        Loop through the cells in the lower surface mesh and calculate the cell area, normal vector, and angle from the normal vector to the free-stream direction.
        """

        lower_surface_mesh = import_lower_surface_mesh()
        cell_areas = calculate_cell_area()
        normal_vectors = calculate_normal_vector()
        angle_from_normal_vector = calculate_angle_from_normal_vector()

        for i in range(len(lower_surface_mesh.vectors)):
            print(f'Cell {i}: Area = {cell_areas[i]}, Normal Vector = {normal_vectors[i]}, Angle from Normal Vector = {angle_from_normal_vector[i]}')
   
    def calculate_cp_modified_newtonian(self, M1):
        """
        Use modified newtonian theory to calculate the pressure distribution given the angle of a unit normal

        Parameters
        ----------
        M1 : float
            Mach number

        Returns
        -------
            - Cp: Pressure distribution given unit normal angle with the freestream
            - post_shock_stagnation_Cp: Cp downstream of the shock
        """
        p_ratio = (1 + (self.gamma - 1) / 2 * M1**2)**(-self.gamma / (self.gamma - 1))

        #Newtonian modified theory
        Cpt = (((p_ratio)*(1+((self.gamma-1)/2)*M1**2)**(self.gamma/(self.gamma-1)))-1)/(0.5*self.gamma*M1**2)
        Cp_newtonian_mod = Cpt*np.cos(self.angles)**2

        return {
            "Cp_newtonian_mod": Cp_newtonian_mod, #Use this one for surface calculations
            "post_shock_stagnation_Cp": Cpt #may be needed in future for forces
        }
    
    def newtonian_pressure_distribution(self):
        """
        Use modified newtonian theory to calculate the pressure distribution given the angle of a unit normal

        Parameters
        ----------
        angles : float
            Angle from the unit normal vector
        """
        
        self.Cp_newtonian = 2*np.sin(self.angles)**2

    def Cp_entire_vehcile(self,Cp_input):
        """
        apply a color to be ascociated with each triangle based on calculated Cp

        Parameters
        ----------
        Cp_input: Array
            Prefered Cp distribution

        Returns
        -------
            - Cp_vehicle: Singular Cp value across entire vehicle. 
        """
        Cp_vehicle = sum(Cp_input*self.cell_areas)/sum(self.cell_areas)

        return Cp_vehicle
        
# Example usage:
if __name__ == "__main__":
    freestream_dir = [1, 0, 0]  # Flow along the x-axis
    analyzer = SurfaceMeshAnalyzer("src/inputs/LowerSurfaceM10-20deg-Meshed.stl")
    analyzer.analyze_mesh(freestream_dir)
    for i in range(len(analyzer.mesh.vectors)):
            print(f'Cell {i}: Area = {analyzer.cell_areas[i]:.6f}, '
                  f'Normal Vector = {analyzer.normal_vectors[i]}, '
                  f'Angle (deg) = {np.degrees(analyzer.angles[i]):.2f}')
