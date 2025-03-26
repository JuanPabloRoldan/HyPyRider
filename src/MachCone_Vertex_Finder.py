import numpy as np
import math

def read_nmb_file(file_ID):
    """
    This function reads and returns the non dimesnionlized chordinates x y z of the left side leading edge.

    Parameters
    ----------
        fileID : .nmb file containing left side leading edge data

    Returns
    ----------
        ref_length : Reference length
        non_dim_chords : 46x3 array of non dimensionlized x y z values
    """

    chords = np.zeros((46, 3))  #Initialize the array for storage
    line_number = 0             #Initialize line number counter

    with open(file_ID, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number >= 8 and line_number <= 53:  #Exclude unwanted data
                str_parts = line.split()                #Split the values into 3 and store in array
                chords[line_number - 8, 0] = float(str_parts[0])
                chords[line_number - 8, 1] = float(str_parts[1])
                chords[line_number - 8, 2] = float(str_parts[2])

    ref_length = np.max(chords[:, 0])     #Obtain reference length
    non_dim_chords = chords / ref_length  #Adjust values

    return{
        "ref_length": ref_length,
        "non_dim_chords": non_dim_chords
    }




def mach_vertex(chords, ref_length):
    """
    This function takes non-dimensionalized chords and runs through Eqn 2.18
    from Bowcutt's dissertation obtaining chords of the Mach cone vertex.

    Parameters
    ----------
    chords : A 46x3 array of non-dimensionalized leading edge data
    ref_length : Reference length

    Returns
    -------
    mach_vertex_chords : A 1x3 array of x, y, z chords for the Mach cone vertex
    """

    x_v = chords[0, 0]     #ID dimensionless chords for vertex
    y_v = -chords[0, 1]
    y_wt = -chords[45, 1]  #And for wingtip
    z_wt = chords[45, 2]


    mu = math.degrees(math.asin(0.1)) 
    c = y_v + x_v * math.degrees(math.tan(mu)) - y_wt  #Obtain c

    #Find X
    x_e_num = c ** 2 + z_wt ** 2 - math.degrees(math.tan(mu)) ** 2
    x_e_dem = 2 * math.degrees(math.tan(mu))  * (c - math.degrees(math.tan(mu)))
    x_e = x_e_num / x_e_dem
    
    #Find y
    y_e = -((1 - x_e) ** 2 * math.degrees(math.tan(mu)) ** 2 - z_wt ** 2) ** 0.5 + y_wt

    #Find z
    z_e = 0

    mach_vertex_chords = np.array([x_e, y_e, z_e]) * ref_length
    
    return{
        "mach_vertex_chords": mach_vertex_chords
    }


# Main execution
file_path = 'LeadingEdgeData_LeftSide.nmb'
try:
    ref_length, dim_chordinates = read_nmb_file(file_path) 
    print(dim_chordinates)

    mach_vertex_chords = mach_vertex(dim_chordinates, ref_length)
    print(mach_vertex_chords)

except Exception as e:
    print(f"Error: {e}")
