import numpy as np
import torch
import torch.nn.functional as F
import open3d
import warp as wp
import os
import sys


_WP_INITIALIZED = False


def _wp_init_once() -> None:
	global _WP_INITIALIZED
	if not _WP_INITIALIZED:
		wp.init()
		_WP_INITIALIZED = True


@wp.kernel
def _raycast_bary_kernel(
	mesh_id: wp.uint64,
	ray_o: wp.array(dtype=wp.vec3),
	ray_d: wp.array(dtype=wp.vec3),
	vtx_pos: wp.array(dtype=wp.vec3),
	vtx_n: wp.array(dtype=wp.vec3),
	out_n: wp.array(dtype=wp.vec3),
	out_depth: wp.array(dtype=wp.float32),
	max_t: float,
	orient_normals_against_ray: int,
):
	tid = wp.tid()

	t = float(0.0)
	u = float(0.0)
	v = float(0.0)
	sign = float(0.0)
	geo_n = wp.vec3(0.0, 0.0, 0.0)
	f = int(0)

	if wp.mesh_query_ray(mesh_id, ray_o[tid], ray_d[tid], max_t, t, u, v, sign, geo_n, f):
		i0 = wp.mesh_get_index(mesh_id, f * 3 + 0)
		i1 = wp.mesh_get_index(mesh_id, f * 3 + 1)
		i2 = wp.mesh_get_index(mesh_id, f * 3 + 2)

		w0 = 1.0 - u - v
		n = w0 * vtx_n[i0] + u * vtx_n[i1] + v * vtx_n[i2]
		n = wp.normalize(n)

		if orient_normals_against_ray != 0:
			if wp.dot(n, ray_d[tid]) > 0.0:
				n = -n

		p = w0 * vtx_pos[i0] + u * vtx_pos[i1] + v * vtx_pos[i2]
		out_n[tid] = n
		out_depth[tid] = wp.length(p - ray_o[tid])
	else:
		out_n[tid] = wp.vec3(0.0, 0.0, 0.0)
		out_depth[tid] = 0.0


def _as_open3d_legacy_triangle_mesh(mesh: object) -> open3d.geometry.TriangleMesh:
	if isinstance(mesh, open3d.geometry.TriangleMesh):
		return mesh

	# Allow Open3D tensor mesh by converting to legacy.
	if hasattr(open3d, "t") and isinstance(mesh, open3d.t.geometry.TriangleMesh):
		return mesh.to_legacy()

	raise TypeError(
		"mesh must be open3d.geometry.TriangleMesh or open3d.t.geometry.TriangleMesh"
	)


def _extract_mesh_arrays_open3d(
	mesh: object,
	device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
	m = _as_open3d_legacy_triangle_mesh(mesh)
	if not m.has_triangles():
		raise ValueError("mesh has no triangles")

	if not m.has_vertex_normals():
		m = open3d.geometry.TriangleMesh(m)
		m.compute_vertex_normals()

	verts_np = np.asarray(m.vertices, dtype=np.float32)
	faces_np = np.asarray(m.triangles, dtype=np.int32).reshape(-1)
	vnorm_np = np.asarray(m.vertex_normals, dtype=np.float32)

	if verts_np.ndim != 2 or verts_np.shape[1] != 3:
		raise ValueError("mesh vertices must be (V,3)")
	if vnorm_np.shape != verts_np.shape:
		raise ValueError("mesh vertex_normals must be (V,3)")
	if faces_np.ndim != 1 or (faces_np.size % 3) != 0:
		raise ValueError("mesh triangles must be (T,3)")

	verts = torch.as_tensor(verts_np, device=device, dtype=torch.float32)
	faces = torch.as_tensor(faces_np, device=device, dtype=torch.int32)
	vnorm = torch.as_tensor(vnorm_np, device=device, dtype=torch.float32)

	return verts.contiguous(), faces.contiguous(), vnorm.contiguous()


class WarpMeshTracer:
	"""CUDA ray-mesh tracing via NVIDIA Warp.

	Produces smooth normals via barycentric interpolation of per-vertex normals.
	Depth is the Euclidean distance from ray origin to the hit point computed
	from barycentric interpolation of triangle vertices.
	"""

	def __init__(
		self,
		verts: torch.Tensor,
		faces_flat: torch.Tensor,
		vertex_normals: torch.Tensor
	):
		_wp_init_once()

		if verts.ndim != 2 or verts.shape[1] != 3:
			raise ValueError("verts must be (V,3)")
		if faces_flat.ndim != 1 or (faces_flat.numel() % 3) != 0:
			raise ValueError("faces_flat must be (T*3,)")
		if vertex_normals.shape != verts.shape:
			raise ValueError("vertex_normals must be (V,3)")
		if not (verts.is_cuda and faces_flat.is_cuda and vertex_normals.is_cuda):
			raise ValueError("verts/faces/vertex_normals must be CUDA tensors")
		if verts.dtype != torch.float32 or vertex_normals.dtype != torch.float32:
			raise ValueError("verts and vertex_normals must be float32")
		if faces_flat.dtype != torch.int32:
			raise ValueError("faces_flat must be int32")

		self.device = f"cuda:{verts.device.index or 0}"

		self._verts_torch = verts.contiguous()
		self._vnorm_torch = vertex_normals.contiguous()
		self._faces_torch = faces_flat.contiguous()

		wp_verts = wp.from_torch(self._verts_torch, dtype=wp.vec3)
		wp_faces = wp.from_torch(self._faces_torch, dtype=wp.int32)

		self._wp_vtx_pos = wp_verts
		self._wp_vtx_n = wp.from_torch(self._vnorm_torch, dtype=wp.vec3)
		self.mesh = wp.Mesh(points=wp_verts, indices=wp_faces)

	@classmethod
	def from_open3d(
		cls,
		mesh: object,
		*,
		device: torch.device | str = "cuda"
	) -> "WarpMeshTracer":
		dev = torch.device(device)
		if dev.type != "cuda":
			raise ValueError("WarpMeshTracer requires a CUDA device")

		verts, faces_flat, vnorm = _extract_mesh_arrays_open3d(mesh, dev)
		return cls(verts, faces_flat, vnorm)

	@torch.no_grad()
	@torch.cuda.nvtx.range("WarpMeshTracer.trace")
	def trace(
		self,
		rays_ori: torch.Tensor,
		rays_dir: torch.Tensor,
		*,
		max_t: float = 1.0e6,
		normalize_dirs: bool = True,
		orient_normals_against_ray: bool = True,
	) -> tuple[torch.Tensor, torch.Tensor]:
		"""Trace rays against the mesh.

		Args:
			rays_ori: (B,H,W,3) float32 CUDA
			rays_dir: (B,H,W,3) float32 CUDA

		Returns:
			normals: (B,H,W,3) float32
			depths:  (B,H,W,1) float32
		"""
		if rays_ori.shape != rays_dir.shape or rays_ori.ndim != 4 or rays_ori.shape[-1] != 3:
			raise ValueError("rays_ori and rays_dir must be (B,H,W,3) with matching shapes")
		if not (rays_ori.is_cuda and rays_dir.is_cuda):
			raise ValueError("rays_ori and rays_dir must be CUDA tensors")
		if rays_ori.dtype != torch.float32 or rays_dir.dtype != torch.float32:
			raise ValueError("rays_ori and rays_dir must be float32")

		B, H, W, _ = rays_ori.shape
		N = B * H * W

		ro = rays_ori.reshape(N, 3).contiguous()
		rd = rays_dir.reshape(N, 3).contiguous()
		if normalize_dirs:
			rd = F.normalize(rd, dim=-1)

		out_n = torch.empty((N, 3), device=ro.device, dtype=torch.float32)
		out_depth = torch.empty((N,), device=ro.device, dtype=torch.float32)

		wp_ro = wp.from_torch(ro, dtype=wp.vec3)
		wp_rd = wp.from_torch(rd, dtype=wp.vec3)
		wp_out_n = wp.from_torch(out_n, dtype=wp.vec3)
		wp_out_depth = wp.from_torch(out_depth, dtype=wp.float32)

		wp.launch(
			_raycast_bary_kernel,
			dim=N,
			inputs=[
				self.mesh.id,
				wp_ro,
				wp_rd,
				self._wp_vtx_pos,
				self._wp_vtx_n,
				wp_out_n,
				wp_out_depth,
				float(max_t),
				1 if orient_normals_against_ray else 0,
			],
			device=self.device,
		)

		normals = out_n.reshape(B, H, W, 3)
		depths = out_depth.reshape(B, H, W, 1)
		return normals, depths


@torch.no_grad()
def trace_rays_against_open3d_mesh(
	mesh: object,
	rays_ori: torch.Tensor,
	rays_dir: torch.Tensor,
	*,
	max_t: float = 1.0e6,
	normalize_dirs: bool = True,
	orient_normals_against_ray: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Convenience wrapper: build a Warp BVH from an Open3D mesh and trace rays.

	If you’re tracing many times against the same mesh, prefer `WarpMeshTracer`
	to avoid rebuilding the BVH.
	"""
	tracer = WarpMeshTracer.from_open3d(mesh, device=rays_ori.device)
	return tracer.trace(
		rays_ori,
		rays_dir,
		max_t=max_t,
		normalize_dirs=normalize_dirs,
		orient_normals_against_ray=orient_normals_against_ray,
	)


def _make_pinhole_rays_cuda(
	*,
	H: int,
	W: int,
	origin: torch.Tensor,
	look_at: torch.Tensor,
	down: torch.Tensor,
	fov_y_degrees: float,
) -> tuple[torch.Tensor, torch.Tensor]:
	"""Create a simple pinhole ray bundle on CUDA.

	Coordinate convention: [right, down, forward] (RDF).

	Returns:
		rays_ori: (1,H,W,3)
		rays_dir: (1,H,W,3)
	"""
	if origin.shape != (3,) or look_at.shape != (3,) or down.shape != (3,):
		raise ValueError("origin/look_at/down must be shape (3,)")
	if origin.device.type != "cuda":
		raise ValueError("origin must be on CUDA")

	forward = (look_at - origin)
	forward = forward / (torch.linalg.norm(forward) + 1e-8)
	down = down / (torch.linalg.norm(down) + 1e-8)

	# Build an orthonormal RDF frame.
	# right = down x forward
	right = torch.cross(down, forward)
	right = right / (torch.linalg.norm(right) + 1e-8)
	# re-orthogonalize down = forward x right
	down = torch.cross(forward, right)
	down = down / (torch.linalg.norm(down) + 1e-8)

	aspect = float(W) / float(H)
	fov_y = float(fov_y_degrees) * (np.pi / 180.0)
	half_h = np.tan(0.5 * fov_y)
	half_w = aspect * half_h

	# Pixel centers in NDC [-1,1]
	js = torch.arange(W, device=origin.device, dtype=torch.float32) + 0.5
	is_ = torch.arange(H, device=origin.device, dtype=torch.float32) + 0.5
	x_ndc = (js / float(W)) * 2.0 - 1.0
	# y increases downward
	y_ndc = (is_ / float(H)) * 2.0 - 1.0
	grid_x, grid_y = torch.meshgrid(x_ndc, y_ndc, indexing="xy")

	# Camera space directions: forward + x*right + y*down
	dirs = forward[None, None, :] + grid_x[..., None] * half_w * right[None, None, :] + grid_y[..., None] * half_h * down[None, None, :]
	dirs = dirs / (torch.linalg.norm(dirs, dim=-1, keepdim=True) + 1e-8)

	rays_ori = origin.view(1, 1, 1, 3).expand(1, H, W, 3).contiguous()
	rays_dir = dirs.view(1, H, W, 3).contiguous()
	return rays_ori, rays_dir


def _write_png(path: str, img_uint8: np.ndarray) -> None:
	if img_uint8.dtype != np.uint8:
		raise ValueError("img_uint8 must be uint8")
	if img_uint8.ndim not in (2, 3):
		raise ValueError("img_uint8 must be HxW or HxWxC")
	os.makedirs(os.path.dirname(path), exist_ok=True)
	o3d_img = open3d.geometry.Image(img_uint8)
	open3d.io.write_image(path, o3d_img)


def _visualize_normals(normals: torch.Tensor) -> np.ndarray:
	"""Map normals [-1,1] -> uint8 RGB."""
	n = normals.detach().float().clamp(-1.0, 1.0)
	n = (n + 1.0) * 0.5
	img = (n * 255.0).round().to(torch.uint8).cpu().numpy()
	return img


def _visualize_depth(depths: torch.Tensor, *, max_t: float) -> np.ndarray:
	"""Map depth to uint8 grayscale with background=0."""
	d = depths.detach().float().squeeze(-1)
	valid = d < (float(max_t) * 0.999)
	if valid.any():
		dv = d[valid]
		d_min = float(dv.min().item())
		d_max = float(dv.max().item())
		den = max(d_max - d_min, 1e-6)
		dn = (d - d_min) / den
		dn = dn.clamp(0.0, 1.0)
	else:
		dn = torch.zeros_like(d)

	# background to 0
	dn = torch.where(valid, dn, torch.zeros_like(dn))
	img = (dn * 255.0).round().to(torch.uint8).cpu().numpy()
	return img