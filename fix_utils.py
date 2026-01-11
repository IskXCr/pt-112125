import numpy as np
import open3d as o3d

# ----------------------------- small linalg helpers -----------------------------

def plane_from_mesh(plane_mesh: o3d.geometry.TriangleMesh):
    V = np.asarray(plane_mesh.vertices, dtype=np.float64)
    if len(V) < 3:
        raise ValueError("Plane mesh must have at least 3 vertices.")
    p0 = V.mean(axis=0)
    X = V - p0
    # normal = eigenvector of smallest variance
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    n = vh[-1]
    n = n / (np.linalg.norm(n) + 1e-12)
    return p0, n

def orient_plane_up_to_object_sign(p0, n, object_pts):
    # Ensure the plane normal points "upward" relative to object cloud
    # (median signed height should be positive)
    h = (object_pts - p0) @ n
    if np.median(h) < 0:
        n = -n
    return n

def orthonormal_basis_from_normal(n):
    n = n / (np.linalg.norm(n) + 1e-12)
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(n @ a) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(n, a)
    u /= (np.linalg.norm(u) + 1e-12)
    v = np.cross(n, u)
    v /= (np.linalg.norm(v) + 1e-12)
    return u, v, n

def bbox_diag(pts):
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    return np.linalg.norm(mx - mn)

# ----------------------------- primitive fitting --------------------------------

def fit_sphere_lstsq(pts):
    # Solve x^2 + y^2 + z^2 + a x + b y + c z + d = 0
    A = np.c_[pts, np.ones((len(pts), 1))]
    b = -(pts[:, 0]**2 + pts[:, 1]**2 + pts[:, 2]**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c, d = sol
    center = -0.5 * np.array([a, b_, c], dtype=np.float64)
    r_sq = np.sum(center**2) - d
    r = float(np.sqrt(max(r_sq, 1e-12)))
    return center, r

def fit_circle_2d_pratt(xy):
    # Solve x^2 + y^2 + a x + b y + c = 0 (least squares)
    A = np.c_[xy, np.ones((len(xy), 1))]
    b = -(xy[:, 0]**2 + xy[:, 1]**2)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_, c = sol
    center = -0.5 * np.array([a, b_], dtype=np.float64)
    r = float(np.sqrt(max(center[0]**2 + center[1]**2 - c, 1e-12)))
    return center, r

# ----------------------------- cap generators -----------------------------------

def make_spherical_cap(center, radius, n, t_join, stacks=48, slices=128):
    """
    center: sphere center (3,)
    radius: sphere radius
    n: unit normal (points 'up')
    t_join: >=0, distance of the cutting plane from center along +n.
            The cap is the part BELOW that plane (toward -n).
    """
    u, v, n = orthonormal_basis_from_normal(n)

    # tip at the bottommost point
    tip = center - n * radius
    verts = [tip]

    mu_max = -t_join / (radius + 1e-12)  # in [-1, 0]
    mu_max = float(np.clip(mu_max, -1.0, 0.0))
    # latitude samples from -1 (bottom) to mu_max (join)
    lat = np.linspace(-1.0, mu_max, stacks + 1)[1:]  # exclude exact -1 row (we have tip)

    for mu in lat:
        z_off = radius * mu
        ring_r = radius * np.sqrt(max(1.0 - mu * mu, 0.0))
        ring_center = center + n * z_off
        ang = np.linspace(0.0, 2.0 * np.pi, num=slices, endpoint=False)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        ring = ring_center[None, :] + ring_r * (cos_a[:, None] * u[None, :] + sin_a[:, None] * v[None, :])
        verts.extend(ring.tolist())

    verts = np.asarray(verts, dtype=np.float64)
    faces = []

    # connect tip to first ring (fan)
    first_ring_start = 1
    for j in range(slices):
        a = 0
        b = first_ring_start + j
        c = first_ring_start + ((j + 1) % slices)
        faces.append([a, b, c])

    # connect ring strips
    for i in range(len(lat) - 1):
        ring0 = 1 + i * slices
        ring1 = 1 + (i + 1) * slices
        for j in range(slices):
            a = ring0 + j
            b = ring0 + ((j + 1) % slices)
            c = ring1 + j
            d = ring1 + ((j + 1) % slices)
            faces.append([a, b, d])
            faces.append([a, d, c])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return mesh

def make_disc(center_on_plane, n, radius, slices=128):
    u, v, n = orthonormal_basis_from_normal(n)
    verts = [center_on_plane.tolist()]
    ang = np.linspace(0.0, 2.0 * np.pi, num=slices, endpoint=False)
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    ring = center_on_plane[None, :] + radius * (cos_a[:, None] * u[None, :] + sin_a[:, None] * v[None, :])
    verts.extend(ring.tolist())
    verts = np.asarray(verts, dtype=np.float64)

    faces = []
    for j in range(slices):
        a = 0
        b = 1 + j
        c = 1 + ((j + 1) % slices)
        # order [a,b,c] so that normal points ~ +n (right-hand rule with u x v = n)
        faces.append([a, b, c])

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(faces, dtype=np.int32))
    return mesh

# ----------------------------- merging ------------------------------------------

def concat_meshes(mesh_a: o3d.geometry.TriangleMesh, mesh_b: o3d.geometry.TriangleMesh):
    Va = np.asarray(mesh_a.vertices, dtype=np.float64)
    Fa = np.asarray(mesh_a.triangles, dtype=np.int32)
    Vb = np.asarray(mesh_b.vertices, dtype=np.float64)
    Fb = np.asarray(mesh_b.triangles, dtype=np.int32)

    V = np.vstack([Va, Vb])
    F = np.vstack([Fa, Fb + len(Va)])

    out = o3d.geometry.TriangleMesh()
    out.vertices = o3d.utility.Vector3dVector(V)
    out.triangles = o3d.utility.Vector3iVector(F)
    return out

# ----------------------------- modes --------------------------------------------

def auto_guess_mode(pts):
    # PCA eigenvalue spread: sphere ~ all equal, glass (cylindrical-ish) -> one small, two large
    X = pts - pts.mean(axis=0)
    C = (X.T @ X) / max(len(X) - 1, 1)
    evals, _ = np.linalg.eigh(C)
    evals = np.sort(evals)
    spread = evals[-1] / max(evals[0], 1e-12)
    # If spread small-ish -> sphere; otherwise -> glass
    return "sphere" if spread < 1.6 else "glass"

# ----------------------------- main function ------------------------------------

def seal_bottom_with_cap(mesh_path: str,
                         plane_mesh_path: str,
                         out_path: str,
                         mode: str = "auto",
                         join_percentile: float = 5.0,
                         band_width_ratio: float = 0.03,
                         cap_stacks: int = 48,
                         cap_slices: int = 128):
    """
    Input:
      mesh_path: path to the broken object (triangle mesh)
      plane_mesh_path: path to the estimated ground plane (triangle mesh)
      out_path: where to write the merged mesh
    Behavior:
      - If mode == "sphere": create a spherical cap and merge.
      - If mode == "glass": create a flat disc and merge.
      - If mode == "auto": choose using PCA on the object.

    Notes:
      - No Poisson reconstruction, no revolving/spin. We simply add the new cap mesh.
      - Works in-place in world coordinates.
    """

    obj = o3d.io.read_triangle_mesh(mesh_path)
    if len(obj.vertices) == 0:
        raise ValueError("Failed to read object mesh.")
    V = np.asarray(obj.vertices, dtype=np.float64)

    plane_mesh = o3d.io.read_triangle_mesh(plane_mesh_path)
    if len(plane_mesh.vertices) == 0:
        raise ValueError("Failed to read plane mesh.")
    p0, n = plane_from_mesh(plane_mesh)
    n = orient_plane_up_to_object_sign(p0, n, V)

    if mode == "auto":
        mode = auto_guess_mode(V)

    diag = bbox_diag(V)
    h = (V - p0) @ n  # signed heights relative to plane
    h_q = np.percentile(h, join_percentile)
    band_tol = max(band_width_ratio * diag, 1e-6)

    # Thin band centered near h_q (instead of everything below)
    band_mask = (h >= h_q - band_tol) & (h <= h_q + band_tol)

    # If band is too small (occlusion/pruning), relax once
    if band_mask.sum() < 100:
        print("Band is too small. Enlarging the band")
        h_q = np.percentile(h, min(join_percentile + 10.0, 50.0))
        band_mask = (h <= h_q + band_tol)
    else:
        print("Accepting the band")

    if mode == "sphere":
        # Fit sphere using points ABOVE the very bottom (avoid jaggies)
        upper_mask = h > np.percentile(h, 35.0)
        if upper_mask.sum() < 200:
            upper_mask = h > np.percentile(h, 20.0)
        c, r = fit_sphere_lstsq(V[upper_mask])

        # Estimate the join circle radius from the lower band points
        band_pts = V[band_mask]
        u, v, n_unit = orthonormal_basis_from_normal(n)
        # planar radial distance from sphere center to band points
        rel = band_pts - c
        a = np.sqrt(np.maximum((rel @ u)**2 + (rel @ v)**2, 0.0))
        a_join = float(np.median(a))  # robust ring radius on the band
        a_join = min(a_join, r * 0.9999)  # clamp

        # Compute t_join so that the circle on the sphere matches a_join
        t_join = float(np.sqrt(max(r * r - a_join * a_join, 0.0)))
        cap = make_spherical_cap(center=c, radius=r, n=n_unit,
                                 t_join=t_join, stacks=cap_stacks, slices=cap_slices)

        merged = concat_meshes(obj, cap)
        o3d.io.write_triangle_mesh(out_path, merged, write_triangle_uvs=False)
        return

    elif mode == "glass":
        # Project lower-band points to the plane and fit a 2D circle -> disc
        band_pts = V[band_mask]
        u, v, n_unit = orthonormal_basis_from_normal(n)

        # --- rim filtering: keep points near the outer radius (reduces side/jagged influence) ---
        rel0 = band_pts - band_pts.mean(axis=0)
        rad = np.sqrt((rel0 @ u)**2 + (rel0 @ v)**2)

        # keep only outermost ring (top 20% radii)
        r_th = np.percentile(rad, 80.0)
        rim_pts = band_pts[rad >= r_th]

        # fallback if too few points
        if rim_pts.shape[0] > 50:
            band_pts = rim_pts
            print("Using the band")

        # local 2D coordinates on plane
        # Use height of band (median) to place the disc on the plane parallel through that height
        hb = (band_pts - p0) @ n_unit

        # Put the disc at the join height (optionally a bit lower to “bite” into the rim)
        disc_height = float(h_q) - 0.25 * band_tol   # try -0.25*band_tol .. -1.0*band_tol
        plane_point = p0 + disc_height * n_unit

        # 2D coordinates relative to plane_point along (u, v)
        rel = band_pts - plane_point
        xy = np.c_[rel @ u, rel @ v]
        if len(xy) < 16:
            # fallback: use all points' projections near the bottom percentile
            fallback_mask = h <= np.percentile(h, 40.0)
            rel_fb = V[fallback_mask] - plane_point
            xy = np.c_[rel_fb @ u, rel_fb @ v]

        (cx, cy), r = fit_circle_2d_pratt(xy)

        r_eff = max(r * 1.08, 1e-6)
        center3d = plane_point + cx * u + cy * v

        disc = make_disc(center_on_plane=center3d, n=n_unit, radius=r_eff, slices=cap_slices)

        merged = concat_meshes(obj, disc)
        o3d.io.write_triangle_mesh(out_path, merged, write_triangle_uvs=False)
        return

    else:
        raise ValueError("mode must be 'auto', 'sphere', or 'glass'")
