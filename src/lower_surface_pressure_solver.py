import numpy
from stl import mesh

def import_lower_surface_mesh():
    """
    Imports the lower surface mesh from the stl file.
    
    Returns:
        lower_surface_mesh (numpy array): The lower surface mesh.
    """
    # Import the pointwise stl file:
    lower_surface_mesh = mesh.Mesh.from_file('LowerSurfaceM10-20deg-Meshed.stl')
    return lower_surface_mesh

def calculate_cell_area():
    """
    Using numpy, calculate the area of each cell in the lower surface mesh.

    """
    # Calculate the area of each cell in the lower surface mesh
    cell_areas = numpy.linalg.norm(numpy.cross(lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 1], lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 2])) / 2
    return cell_areas

def calculate_normal_vector():
    """
    Using numpy, calculate the normal vector of each cell in the lower surface mesh.

    """
    # Calculate the normal vector of each cell in the lower surface mesh
    normal_vectors = numpy.cross(lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 1], lower_surface_mesh.vectors[:, 0] - lower_surface_mesh.vectors[:, 2])
    return normal_vectors

def calculate_angle_from_normal_vector():
    """
    Using numpy, calculate the angle from the normal vector to the free-stream direction.

    """
    # Calculate the angle from the normal vector to the free-stream direction
    angle_from_normal_vector = numpy.arccos(numpy.dot(normal_vectors, [0, 0, 1]) / (numpy.linalg.norm(normal_vectors) * numpy.linalg.norm([0, 0, 1])))
    return angle_from_normal_vector

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