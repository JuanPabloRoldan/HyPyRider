import os
import numpy as np
import pandas as pd
from conical_flow_analyzer import ConicalFlowAnalyzer
import process_LE_points

class StreamlineIntegrator:
    def __init__(self, gamma, M1, theta_s):
        """
        Initializes the streamline integrator.
        This includes creating an instance of the conical flow solver,
        as well as saving a data frame of useful solutions from the conical shock
        solutions from theta = [shock angle, cone angle].

        Parameters:
            gamma (float): Specific heat ratio for the fluid.
            M1 (float): Freestream Mach number upstream of the shock.
            theta_s (float): Shock wave angle in radians.
        """
        self.gamma = gamma
        self.M1 = M1
        self.theta_s = theta_s
        self.streamline_data = [] # store streamline points with ID

        # Create an instance of ConicalFlowAnalyzer
        self.conical_analyzer = ConicalFlowAnalyzer(M1, gamma)
        self.theta_c, _, _ = self.conical_analyzer.solve_taylor_maccoll(self.theta_s)

        # Tabulate post-shock flow properties from shock angle to the cone angle
        self.TM_tabulation = self.conical_analyzer.tabulate_tm_shock_to_cone(theta_s)

    def trace_streamline(self, x, y, z, streamline_id):
        """
        Traces a streamline starting from the given leading-edge (LE) normalized coordinates.

        Parameters:
            x (float): Normalized x-coordinate of the LE point.
            y (float): Normalized y-coordinate of the LE point.
            z (float): Normalized z-coordinate of the LE point.
            streamline_id (int): Unique ID for this streamline.

        Returns:
            None (stores streamline points in self.streamline_data)
        """
        theta = self.theta_s
        streamline_points = []
        order = 0 # tracks # of points in a streamline

        # Store the initial point
        streamline_points.append([x * self.ref_length, 
                                  y * self.ref_length, 
                                  z * self.ref_length, streamline_id, order])

        order +=1

        while x < self.ref_length:
            
            r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            alpha = np.arctan(abs(z / y))
            dt = 0.02

            # Interpolate V_r and V_theta from tabulated data
            V_r = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_r'])
            V_theta = np.interp(theta, self.TM_tabulation['Theta (radians)'], self.TM_tabulation['V_theta'])

            # Update theta and r
            d_theta = V_theta * dt / r
            theta += d_theta
            r += (V_r * dt)
            w = np.sqrt(y ** 2 + z ** 2)

            # Update coordinates
            x = r * np.cos(theta)
            y = -w * np.cos(alpha)
            z = w * np.sin(alpha)

            # Store points with streamline ID and order
            streamline_points.append([x * self.ref_length, 
                                      y * self.ref_length, 
                                      z * self.ref_length, streamline_id, order])
            
            order += 1

            if np.isclose(theta, self.theta_c, rtol=0.01):
                break

        # Append streamline points to global data
        self.streamline_data.extend(streamline_points)

    def create_lower_surface(self):
        """
        Generates the lower surface by tracing streamlines from the leading-edge (LE) points.

        The function normalizes the LE points, traces each streamline, and prints the 
        trajectory to the console.

        Parameters:
            None

        Returns:
            None
        """

        # Find the maximum x-coordinate to normalize the lengths
        self.ref_length = self.LE_points['X'].max()

        # Iterate over all LE points
        for index, row in self.LE_points.iterrows():

            # Grab points and normalize lengths
            x, y, z = row['X'], row['Y'], row['Z']
            x /= self.ref_length
            y /= self.ref_length
            z /= self.ref_length

            # Trace the streamline for the current LE point
            self.trace_streamline(x, y, z, streamline_id=index)

    def export_streamlines_vtk(self, filename):
        """
        Exports the streamlines to a VTK file for visualization.

        Parameters:
            filename (str): Name of the output VTK file.
        
        Returns:
            None
        """
        # Ensure the output directory exists
        output_dir = "src/outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write('Streamline Data\n')
            f.write("ASCII\n")
            f.write("DATASET POLYDATA\n")

            # Write all points
            f.write(f"POINTS {len(self.streamline_data)} float\n")
            for p in self.streamline_data:
                f.write(f"{p[0]} {p[1]} {p[2]}\n")
            
            # Write connectivity (polylines for each streamline)
            streamline_ids = list(set([p[3] for p in self.streamline_data]))  # Unique streamline IDs
            f.write(f"\nLINES {len(streamline_ids)} {len(self.streamline_data) + len(streamline_ids)}\n")

            last_id = -1
            line_points = []
            for i, p in enumerate(self.streamline_data):
                if p[3] != last_id:
                    if last_id != -1:
                        f.write(f"{len(line_points)} " + " ".join(map(str, line_points)) + "\n")
                    last_id = p[3]
                    line_points = []
                line_points.append(i)

            # Write last streamline
            if line_points:
                f.write(f"{len(line_points)} " + " ".join(map(str, line_points)) + "\n")

        print(f"Streamline data saved to {filename}")

    def export_streamlines_dat(self, filename):
        """
        Exports the streamlines to a .dat file in a column format (x y z streamline_id order),
        with tab delimiters and the number of points at the top of each segment.

        Parameters:
            filename (str): Name of the output .dat file.

        Returns:
            None
        """
        output_dir = "src/outputs"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            # Sort streamline data by ID and order to ensure proper ordering
            self.streamline_data.sort(key=lambda p: (p[3], p[4]))
            
            # Extract unique streamline IDs
            streamline_ids = sorted(set(point[3] for point in self.streamline_data))
            
            # Store first and last points to create connectivity segments
            first_points = []
            last_points = []

            for streamline_id in streamline_ids:
                streamline = [p for p in self.streamline_data if p[3] == streamline_id]

                # Store the first and last points
                first_points.append(streamline[0])
                last_points.append(streamline[-1])

                # Write the number of points in the streamline
                f.write(f"{len(streamline)}\n")

                # Write the coordinates with streamline_id and order
                for x, y, z, s_id, order in streamline:
                    f.write(f"{x}\t{y}\t{z}\t{s_id}\t{order}\n")

            # Write connectivity segments between streamlines
            f.write("\n# Connectivity segments between streamlines\n")
            
            for i in range(len(first_points) - 1):
                # First point to first point connection
                x1, y1, z1, s_id1, order1 = first_points[i]
                x2, y2, z2, s_id2, order2 = first_points[i + 1]
                f.write(f"2\n")
                f.write(f"{x1}\t{y1}\t{z1}\t{s_id1}\t{order1}\n")
                f.write(f"{x2}\t{y2}\t{z2}\t{s_id2}\t{order2}\n")

                # Last point to last point connection
                x1, y1, z1, s_id1, order1 = last_points[i]
                x2, y2, z2, s_id2, order2 = last_points[i + 1]
                f.write(f"2\n")
                f.write(f"{x1}\t{y1}\t{z1}\t{s_id1}\t{order1}\n")
                f.write(f"{x2}\t{y2}\t{z2}\t{s_id2}\t{order2}\n")

        print(f"Streamline segment data saved to {filename}")

# Example Usage
if __name__ == "__main__":
    # Initialize the streamline integrator with specific parameters
    integrator = StreamlineIntegrator(gamma=1.4, M1=10.0, theta_s=np.radians(20))

    # Extract leading-edge points from a file
    file_path = 'src/inputs/LeadingEdgeData_LeftSide.nmb'
    integrator.LE_points = process_LE_points.extract_points_from_file(file_path)

    # Create the lower surface by tracing streamlines
    integrator.create_lower_surface()

    integrator.export_streamlines_dat("streamlines.dat")

    # Export streamlines for ParaView visualization
    integrator.export_streamlines_vtk("streamlines.vtk")