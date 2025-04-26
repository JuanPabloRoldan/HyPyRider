import os

def shift_y_coords_in_nmb(input_file, output_file, y_shift=4.55):
    new_lines = []
    in_curve = False

    with open(input_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        stripped = line.strip()

        # Detect start and end of BezierCurve section
        if "<BezierCurve" in stripped:
            in_curve = True
        elif "</BezierCurve>" in stripped:
            in_curve = False

        if in_curve:
            # Try to parse lines that look like coordinates
            parts = stripped.split()

            if len(parts) == 3:
                try:
                    x, y, z = map(float, parts)
                    # Shift the y-coordinate
                    y += y_shift
                    new_line = f"{x:.16f} {y:.16f} {z:.16f}\n"
                    new_lines.append(new_line)
                    continue
                except ValueError:
                    # Not a coordinate line; fall back to writing original line
                    pass

        # Otherwise, keep original line
        new_lines.append(line)

    # Write out the new file
    with open(output_file, "w") as f:
        f.writelines(new_lines)

    print(f"Shifted file written to {output_file}")

# === Example usage ===
input_file = "src/inputs/right-expansion-side.nmb"
output_file = "src/inputs/right-expansion-side-shifted.nmb"

shift_y_coords_in_nmb(input_file, output_file, y_shift=4.55)
