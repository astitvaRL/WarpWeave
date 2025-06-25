from animation.prepare_animation_data import AnimationData

import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.sim
import warp.sim.render
import time
import trimesh
import igl
import pymeshlab
from scipy import spatial
from umbra import MeshViewer
import copy

class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value


def get_boundary_verts_pos(meshset):
    ms = meshset
    all_verts = ms.current_mesh().vertex_matrix()
    ms.compute_selection_from_mesh_border()
    ms.apply_selection_inverse(invfaces=True, invverts=True)
    ms.meshing_remove_selected_vertices()
    boundary_verts = ms.current_mesh().vertex_matrix()
    ymean = np.mean(boundary_verts[:, 1])
    boundary_verts_filtered = boundary_verts[boundary_verts[:, 1] > ymean, :]

    return boundary_verts_filtered



def load_obj_mesh(file_path):
    """
    Load an OBJ mesh file and return vertices and indices.

    Args:
        file_path: Path to the OBJ file

    Returns:
        vertices: List of vertex positions as wp.vec3
        indices: List of triangle indices
    """
    vertices = []
    indices = []

    with open(file_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue

        if parts[0] == 'v':
            # Vertex position
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            vertices.append(wp.vec3(x, y, z))
        elif parts[0] == 'f':
            # Face indices (OBJ uses 1-based indexing)
            # Handle different face formats (v, v/vt, v/vt/vn, v//vn)
            face_vertices = []
            for part in parts[1:]:
                vertex_index = int(part.split('/')[0]) - 1  # Convert to 0-based indexing
                face_vertices.append(vertex_index)

            # Triangulate faces if they have more than 3 vertices
            if len(face_vertices) == 3:
                indices.extend(face_vertices)
            elif len(face_vertices) > 3:
                # Simple triangulation for convex polygons
                for i in range(1, len(face_vertices) - 1):
                    indices.extend([face_vertices[0], face_vertices[i], face_vertices[i + 1]])

    return vertices, indices

# Kernel to apply a sine wave deformation to the body mesh
@wp.kernel
def update_cloth_pins_kernel(
    particles: wp.array(dtype=wp.vec3f),
    particle_flags: wp.array(dtype=wp.uint32)
):
    tid = wp.tid()
    if particle_flags[tid] == 0:
        particles[tid] = particles[tid] + wp.vec3f(0.0, 0.0, 0.0)

class DataSimulator:
    def __init__(
        self,
        body_verts,
        body_faces,
        body_indices,
        cloth_verts,
        cloth_faces,
        cloth_indices,
        body_points_array_set=[],
        stage_path="data_sim.usd",
        integrator: IntegratorType = IntegratorType.VBD,
        scale=1.0,
        body_scale=1.0,
        position=wp.vec3(0.0, 0.0, 0.0),
        rotation=wp.quat_identity(),
        body_position=wp.vec3(0.0, 0.0, 0.0),
        body_rotation=wp.quat_identity(),
        density=0.1,  # density for cloth mesh
        total_steps=None,
        update_start_step_num = -1,
        final_relax_steps = 0,
    ):
        self.integrator_type = integrator

        fps = 60
        self.sim_substeps = 16
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.profiler = {}
        self.total_steps = total_steps
        self.update_start_step_num = update_start_step_num
        if self.update_start_step_num > -1:
            self.update_end_step_num = self.update_start_step_num + len(body_points_array_set)
        self.final_relax_steps = final_relax_steps

        builder = wp.sim.ModelBuilder()

        # If no OBJ path is provided, use a default cloth grid

        self.cloth_vertices = cloth_verts
        self.cloth_indices = cloth_indices
        self.cloth_faces = cloth_faces

        self.body_vertices = body_verts
        self.body_indices = body_indices
        self.body_faces = body_faces
        self.body_points_array_set = body_points_array_set
        if len(self.body_points_array_set) > 0:
            self.body_points_array_set = self.body_points_array_set * body_scale

        # Create a new meshset in pymeshlab from vertices and faces
        mesh_set = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=np.array(self.cloth_vertices), face_matrix=np.array(self.cloth_faces))
        mesh_set.add_mesh(mesh)

        # estimate pinning vertices
        boundary_verts = get_boundary_verts_pos(mesh_set)
        tree = spatial.KDTree(np.array(self.cloth_vertices))
        dist, idx = tree.query(boundary_verts, k=1, p=2)
        pinning_mask = np.array([1]*len(self.cloth_vertices))
        pinning_mask[idx] = 0
        pinning_mask = pinning_mask.tolist()
        self.pinning_mask = wp.array(pinning_mask, dtype=wp.uint32)

        # Scale vertices if needed
        self.cloth_vertices = [wp.vec3(v[0] * scale, v[1] * scale, v[2] * scale) for v in self.cloth_vertices]

        # # # Add cloth mesh to the simulation
        # builder.add_cloth_mesh(
        #     pos=position,
        #     rot=rotation,
        #     scale=1.0,  # Already scaled the vertices if needed
        #     vertices=self.cloth_vertices,
        #     indices=self.cloth_indices,
        #     vel=wp.vec3(0.0, 0.0, 0.0),
        #     add_springs=True,  # Add triangle bending springs
        #     spring_ke=1.2e3,
        #     spring_kd=1.0,
        #     particle_radius=0.5,
        #     density=0.5,  # Use the provided density as density
        #     tri_ke=1.0e-2,  # Triangle stretch stiffness
        #     tri_ka=1.0e6-2,  # Triangle area stiffness
        #     tri_kd=1.0e-5,  # Triangle damping
        #     edge_ke=1.0e-1,   # Edge stretch stiffness
        #     edge_kd=10.0,   # Edge bending stiffness
        # )

        builder.add_cloth_mesh(
            pos=position,
            rot=rotation,
            scale=1.0,  # Already scaled the vertices if needed
            vertices=self.cloth_vertices,
            indices=self.cloth_indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            add_springs=True,  # Add triangle bending springs
            spring_ke=5.0e3,
            spring_kd=10.0,
            particle_radius=0.5,
            density=1.0,  # Use the provided density as density
            tri_ke=1.0,  # Triangle stretch stiffness
            tri_ka=1.0,  # Triangle area stiffness
            tri_kd=10.0,  # Triangle damping
            edge_ke=1.0,   # Edge stretch stiffness
            edge_kd=10.0,   # Edge bending stiffness
        )


        self.body_vertices = [wp.vec3(v[0] * body_scale, v[1] * body_scale, v[2] * body_scale) for v in self.body_vertices]

        # Convert to numpy arrays for wp.sim.Mesh
        body_points = np.array([[v[0], v[1], v[2]] for v in self.body_vertices])
        body_indices_np = np.array(self.body_indices)

        self.body_mesh = wp.sim.Mesh(body_points, body_indices_np)

        self.body_mesh_id = builder.add_shape_mesh(
            body=-1,
            mesh=self.body_mesh,
            pos=body_position,
            rot=body_rotation,
            scale=wp.vec3(1.0, 1.0, 1.0),  # Already scaled the vertices if needed
        )

        if self.integrator_type == IntegratorType.VBD:
            builder.color()

        self.model = builder.finalize()
        self.body_points_array = self.model.shape_geo_src[self.body_mesh_id].mesh.points

        self.model.particle_flags = self.pinning_mask
        self.model.ground = False  # Enable ground plane for collision
        self.model.enable_tri_collisions = True
        self.model.enable_particle_particle_collisions = True
        self.model.enable_triangle_particle_collisions = True
        self.model.enable_edge_edge_collisions = True

        # set up contact query and contact detection distances
        self.model.soft_contact_ke = 1.0e5
        self.model.soft_contact_kd = 1.0e-6
        self.model.soft_contact_mu = 0.2
        self.model.soft_contact_radius = 0.5
        self.model.soft_contact_margin = 0.5

        if self.integrator_type == IntegratorType.EULER:
            self.integrator = wp.sim.SemiImplicitIntegrator()
        elif self.integrator_type == IntegratorType.XPBD:
            self.integrator = wp.sim.XPBDIntegrator(iterations=1)
        else:
            self.integrator = wp.sim.VBDIntegrator(
                self.model,
                iterations=1,
                handle_self_contact=True
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=100.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def update_body_mesh(self, step_num):
        # indexing into numpy array set first and then convert to wp.array, as we can't index into a wp.array set directly
        self.model.shape_geo_src[self.body_mesh_id].mesh.points = wp.array(self.body_points_array_set[step_num], dtype=wp.vec3f)

    def update_cloth_pins(self, step_num):
         wp.launch(
            kernel=update_cloth_pins_kernel,
            dim=len(self.state_0.particle_q),
            inputs=[
                self.state_0.particle_q,
                self.model.particle_flags]
            )

    def simulate(self):
        wp.sim.collide(self.model, self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()

            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)

            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self, step_num):
        if self.update_start_step_num>-1 and step_num > self.update_start_step_num and step_num < self.update_end_step_num:
            step_offset = step_num - self.update_start_step_num
            # Update the body mesh and cloth pins before simulation step
            self.update_body_mesh(step_offset)
            # self.update_cloth_pins(step_offset)
        with wp.ScopedTimer("step", dict=self.profiler):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render"):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.render_mesh(
                name = "body",
                points = self.model.shape_geo_src[self.body_mesh_id].mesh.points.numpy(),
                indices = self.model.shape_geo_src[self.body_mesh_id].mesh.indices.numpy()
            )
            self.renderer.end_frame()

    def cleanup(self):
        """
        Clean up resources used by the simulator.
        Call this method when you're done with the simulator to free up resources.
        """
        # Release CUDA graph if it exists
        if hasattr(self, 'graph') and self.graph is not None:
            self.graph = None

        # Save and close the renderer if it exists
        if self.renderer is not None:
            try:
                self.renderer.save()
            except Exception as e:
                print(f"Warning: Could not save renderer: {e}")
            self.renderer = None

        # Clear large data arrays
        if hasattr(self, 'body_points_array_set'):
            self.body_points_array_set = None

        # Force garbage collection to clean up GPU memory
        import gc
        gc.collect()

        # Explicitly synchronize the device to ensure all operations are complete
        wp.synchronize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="datasim_cloth_mesh.usd",
        help="Name of the output USD file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save the output USD file.",
    )
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=IntegratorType.XPBD,
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the loaded mesh.",
    )
    parser.add_argument(
        "--position",
        type=float,
        nargs=3,
        default=[0.0, 4.0, 0.0],
        help="Initial position of the cloth mesh (x, y, z).",
    )
    parser.add_argument(
        "--body_scale",
        type=float,
        default=1.0,
        help="Scale factor to apply to the loaded body mesh.",
    )
    parser.add_argument(
        "--body_position",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="Initial position of the body mesh (x, y, z).",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.1,
        help="density of cloth particles.",
    )

    args = parser.parse_known_args()[0]

    if args.out_dir is None:
        args.out_dir = f"C:\\Users\\astitva\\Desktop\\warp_out\\{str(time.time()).split('.')[0]}"

    # Create output directory if it doesn't exist
    if args.out_dir and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # prepare existing simulation data
    usd_load_dir = "D:\\Simulation\\DATA\\PhysRig\\skirt\\usd\\"
    usd_files = os.listdir(usd_load_dir)

    sim_data = dict()
    for usd_filename in usd_files:
        stage = Usd.Stage.Open(f'{usd_load_dir}\\{usd_filename}')
        for prim in stage.TraverseAll():
            if prim.GetPrimPath().name.endswith("surface"):
                geomesh = UsdGeom.Mesh(prim)
                timesamples = geomesh.GetPointsAttr().GetTimeSamples()
                maxtime = max(timesamples)
                c_verts = np.array(geomesh.GetPointsAttr().Get(maxtime))
            if prim.GetPrimPath().name.endswith("body"):
                geomesh = UsdGeom.Mesh(prim)
                timesamples = geomesh.GetPointsAttr().GetTimeSamples()
                maxtime = max(timesamples)
                b_verts = np.array(geomesh.GetPointsAttr().Get(maxtime))
        sim_data[usd_filename.split('_')[-1][:-4]] =  {'cloth': c_verts, 'body': b_verts}

    # just to read faces and indices
    load_dir = "D:\\Simulation\\hermes\\hermes_local_data\\interpolated_data\\"
    anim = AnimationData(load_dir)
    anim.load_data()
    _, body_faces, body_indices = anim.get_body_mesh()
    _, cloth_faces, cloth_indices = anim.get_cloth_mesh()

    # viewer = MeshViewer()

    for pose_idx in range(len(usd_files)):

        print("Simulating pose: ", pose_idx)
        cloth_verts = sim_data[usd_files[pose_idx].split('_')[-1][:-4]]['cloth']
        body_verts = sim_data[usd_files[pose_idx].split('_')[-1][:-4]]['body']

        with wp.ScopedDevice(args.device):
            full_stage_path = os.path.join(args.out_dir, f'{usd_files[pose_idx]}')
            datasim = DataSimulator(
                body_verts=body_verts.copy(),
                body_faces=body_faces.copy(),
                body_indices=body_indices.copy(),
                cloth_verts=cloth_verts.copy(),
                cloth_faces=cloth_faces.copy(),
                cloth_indices=cloth_indices.copy(),
                stage_path=full_stage_path,
                integrator=args.integrator,
                scale=args.scale,
                body_scale=args.body_scale,
                position=wp.vec3(args.position[0], args.position[1], args.position[2]),
                body_position=wp.vec3(args.body_position[0], args.body_position[1], args.body_position[2]),
                density=args.density,
                )
            for _i in range(1000):
                datasim.step(step_num=_i)
                print(f"Frame {_i+1}")
                # cloth_verts = datasim.state_0.particle_q.numpy()
                # cloth_faces = np.array(datasim.cloth_indices).reshape(-1,3)
                # cloth_colors = np.ones_like(cloth_verts) * np.array([0.3, 0.5, 0.55])
                # body_verts = datasim.model.shape_geo_src[datasim.body_mesh_id].mesh.points.numpy()
                # body_faces = datasim.model.shape_geo_src[datasim.body_mesh_id].mesh.indices.numpy().reshape(-1,3)
                # body_colors = np.ones_like(body_verts) * np.array([0.3, 0.3, 0.3])
                # viewer.set_mesh(v=cloth_verts, f=cloth_faces, c=cloth_colors, object_name="cloth")
                # viewer.set_mesh(v=body_verts, f=body_faces, c=body_colors, object_name="body")

        # breakpoint()
        datasim.render()
        datasim.renderer.save()

        datasim.cleanup()
        del datasim
