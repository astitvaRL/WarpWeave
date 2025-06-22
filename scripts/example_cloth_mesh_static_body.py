import math
import os
from enum import Enum

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render
import time
import trimesh
import igl
import pymeshlab
from scipy import spatial

class IntegratorType(Enum):
    EULER = "euler"
    XPBD = "xpbd"
    VBD = "vbd"

    def __str__(self):
        return self.value


def get_boundary_verts_pos(obj_path):
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
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


class Example:
    def __init__(
        self,
        stage_path="example_cloth_mesh.usd",
        integrator: IntegratorType = IntegratorType.VBD,
        obj_path=None,
        body_obj_path=None,
        scale=1.0,
        body_scale=1.0,
        position=wp.vec3(0.0, 0.0, 0.0),
        rotation=wp.quat_identity(),
        body_position=wp.vec3(0.0, 0.0, 0.0),
        body_rotation=wp.quat_identity(),
        density=0.1,  # density for cloth mesh
        total_steps=None,
    ):
        self.integrator_type = integrator

        fps = 60
        self.sim_substeps = 16
        self.frame_dt = 1.0 / fps
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.sim_time = 0.0
        self.profiler = {}

        builder = wp.sim.ModelBuilder()

        # If no OBJ path is provided, use a default cloth grid

        print(f"Loading cloth mesh from {obj_path}")
        # Load the OBJ mesh
        vertices, indices = load_obj_mesh(obj_path)
        print("Cloth mesh loaded successfully!")

        # estimate pinning vertices
        boundary_verts = get_boundary_verts_pos(obj_path)
        tree = spatial.KDTree(np.array(vertices))
        dist, idx = tree.query(boundary_verts, k=1, p=2)
        pinning_mask = np.array([1]*len(vertices))
        pinning_mask[idx] = 0
        pinning_mask = pinning_mask.tolist()
        pinning_mask = wp.array(pinning_mask, dtype=wp.uint32)

        # Scale vertices if needed
        if scale != 1.0:
            vertices = [wp.vec3(v[0] * scale, v[1] * scale, v[2] * scale) for v in vertices]

        # # Add cloth mesh to the simulation
        builder.add_cloth_mesh(
            pos=position,
            rot=rotation,
            scale=1.0,  # Already scaled the vertices if needed
            vertices=vertices,
            indices=indices,
            vel=wp.vec3(0.0, 0.0, 0.0),
            add_springs=True,  # Add triangle bending springs
            spring_ke=5.0e3,
            spring_kd=10.0,
            particle_radius=0.05,
            density=1.0,  # Use the provided density as density
            tri_ke=1.0e4,  # Triangle stretch stiffness
            tri_ka=1.0e4,  # Triangle area stiffness
            # tri_kd=1.0e-5,  # Triangle damping
            edge_ke=1.0,   # Edge stretch stiffness
            edge_kd=10,   # Edge bending stiffness
        )

        # Add a body mesh as a collision object if provided, otherwise use the bunny mesh
        print(f"Loading body mesh from {body_obj_path}")
        # Load the body OBJ mesh
        body_vertices, body_indices = load_obj_mesh(body_obj_path)
        print("Body mesh loaded successfully!")

        body_vertices = [wp.vec3(v[0] * body_scale, v[1] * body_scale, v[2] * body_scale) for v in body_vertices]

        # Convert to numpy arrays for wp.sim.Mesh
        body_points = np.array([[v[0], v[1], v[2]] for v in body_vertices])
        body_indices_np = np.array(body_indices)

        self.body_mesh = wp.sim.Mesh(body_points, body_indices_np)

        self.body_mesh_id = builder.add_shape_mesh(
            body=-1,
            mesh=self.body_mesh,
            pos=body_position,
            rot=body_rotation,
            scale=wp.vec3(1.0, 1.0, 1.0),  # Already scaled the vertices if needed
            ke=1.0e2,
            kd=1.0e2,
            kf=1.0e1,
        )

        if self.integrator_type == IntegratorType.VBD:
            builder.color()

        self.model = builder.finalize()

        self.model.particle_flags = pinning_mask
        self.model.ground = False  # Enable ground plane for collision
        self.model.enable_tri_collisions = True
        self.model.enable_particle_particle_collisions = True
        self.model.enable_triangle_particle_collisions = True
        self.model.enable_edge_edge_collisions = True


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
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=40.0)
        else:
            self.renderer = None

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def simulate(self):
        wp.sim.collide(self.model, self.state_0)
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            # swap states
            (self.state_0, self.state_1) = (self.state_1, self.state_0)

    def step(self, step_num=None):
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
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="example_cloth_mesh.usd",
        help="Name of the output USD file.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument(
        "--integrator",
        help="Type of integrator",
        type=IntegratorType,
        choices=list(IntegratorType),
        default=IntegratorType.XPBD,
    )
    parser.add_argument(
        "--obj_path",
        type=str,
        default=None,
        help="Path to the OBJ mesh file to use as cloth.",
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
        "--body_obj_path",
        type=str,
        default=None,
        help="Path to the OBJ mesh file to use as body.",
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
    parser.add_argument(
        "--render_freq",
        type=int,
        default=0,
        help="interval between renders, 1 for rendering all frames, 0 for no render",
    )


    args = parser.parse_known_args()[0]

    if args.out_dir is None:
        args.out_dir = f"C:\\Users\\astitva\\Desktop\\warp_out\\{str(time.time()).split('.')[0]}"

    # Create output directory if it doesn't exist
    if args.out_dir and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # Combine output directory with stage path
    full_stage_path = os.path.join(args.out_dir, args.stage_path) if args.out_dir else args.stage_path

    with wp.ScopedDevice(args.device):
        example = Example(
            stage_path=full_stage_path,
            integrator=args.integrator,
            obj_path=args.obj_path,
            body_obj_path=args.body_obj_path,
            scale=args.scale,
            body_scale=args.body_scale,
            position=wp.vec3(args.position[0], args.position[1], args.position[2]),
            body_position=wp.vec3(args.body_position[0], args.body_position[1], args.body_position[2]),
            density=args.density,
            total_steps=args.num_frames
        )

        for _i in range(args.num_frames):
            example.step(step_num=_i)
            if args.render_freq>0:
                if _i%args.render_freq == 0:
                    print("rendering frame")
                    example.render()
            print(f"Frame {_i+1}/{args.num_frames}")

        if args.render_freq==0: # render the latest (last) frame
            example.render()

        frame_times = example.profiler["step"]
        print("\nAverage frame sim time: {:.2f} ms".format(sum(frame_times) / len(frame_times)))

        if example.renderer:
            example.renderer.save()
