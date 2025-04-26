import numpy as np

def bilinear_interpolate_moc(eta, xi, flow_fields):
    """
    Performs bilinear interpolation of flow features in the MoC grid.

    Parameters:
        eta, xi (float): Logical coordinates
        flow_fields (dict): Dictionary with 2D arrays:
            - 'u_field', 'v_field' (same shape)
            - 'M_field', 'q_field', etc. if needed

    Returns:
        dict: Interpolated values, e.g. {"u": u_interp, "v": v_interp}
    """
    u_field = flow_fields["u_field"]
    v_field = flow_fields["v_field"]

    i = int(np.floor(eta))
    j = int(np.floor(xi))

    # Ensure we're inside bounds
    if i < 0 or i >= u_field.shape[0]-1 or j < 0 or j >= u_field.shape[1]-1:
        return None

    t = eta - i
    s = xi - j

    # Bilinear interpolation
    def bilinear(f):
        return (
            f[i, j]     * (1 - t) * (1 - s) +
            f[i+1, j]   * t       * (1 - s) +
            f[i, j+1]   * (1 - t) * s +
            f[i+1, j+1] * t       * s
        )

    return {
        "u": bilinear(u_field),
        "v": bilinear(v_field)
    }