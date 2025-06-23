import torch
import numpy as np
import os
import pymeshlab

class AnimationData:
    def __init__(self, load_dir):
        self.load_dir = load_dir
        self.data_dict = {}

    def load_data(self):
        for filename in os.listdir(self.load_dir):
            if filename.endswith(".pt"):
                data = torch.load(self.load_dir + filename, weights_only=False)
                self.data_dict[filename.split(".")[0]] = data
        return self.data_dict

    def get_animation_body_points(self):
        return self.data_dict['all_pose_b_interpolated']['all_pose_b_interp'].numpy()

    def get_body_mesh(self):
        vertices = self.data_dict['scene']['scene']['bodyPos']
        faces =  self.data_dict['scene']['scene']['bodyFaces']
        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=np.array(vertices), face_matrix=np.array(faces))
        ms.add_mesh(mesh, 'body')
        ms.meshing_invert_face_orientation()
        ms.apply_normal_normalization_per_face()
        ms.compute_normal_per_vertex()
        ms.apply_normal_normalization_per_vertex()
        ms.save_current_mesh('temp_body.obj')
        v = ms.current_mesh().vertex_matrix()
        f = ms.current_mesh().face_matrix()
        i = f.flatten()
        return v, f, i

    def get_cloth_mesh(self):
        vertices = self.data_dict['scene']['scene']['initPos']
        faces = self.data_dict['scene']['scene']['faces']
        ms = pymeshlab.MeshSet()
        mesh = pymeshlab.Mesh(vertex_matrix=np.array(vertices), face_matrix=np.array(faces))
        ms.add_mesh(mesh, 'cloth')
        ms.meshing_invert_face_orientation()
        ms.apply_normal_normalization_per_face()
        ms.compute_normal_per_vertex()
        ms.apply_normal_normalization_per_vertex()
        ms.meshing_surface_subdivision_loop()
        ms.save_current_mesh('temp_cloth.obj')
        v = ms.current_mesh().vertex_matrix()
        f = ms.current_mesh().face_matrix()
        i = f.flatten()
        return v, f, i
