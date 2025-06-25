import bpy
import os

# --- User configuration ---
blend_file = 'D:\\Simulation\\DATA\\Bender_Stuff\\BLEND_FILES\\usd_render.blend'
usd_folder = 'D:\\Simulation\\DATA\\PhysRig\\skirt\\usd_refined\\'
output_folder = 'D:\\Simulation\\DATA\\Bender_Stuff\\RENDERS\\usd_render_refined\\'
os.makedirs(output_folder, exist_ok=True)
# --------------------------

# Open the specified .blend file
bpy.ops.wm.open_mainfile(filepath=blend_file)

# Set the render engine to Workbench
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 1080
bpy.context.scene.render.resolution_y = 1080

# Get all USD files in the folder
usd_files = sorted([f for f in os.listdir(usd_folder) if f.endswith('.usd')])

for usd_file in usd_files:
    usd_path = os.path.join(usd_folder, usd_file)
    # Import the USD file
    bpy.ops.wm.usd_import(filepath=usd_path)

    # Deselect all objects
    bpy.ops.object.select_all(action='DESELECT')

    # Select only "body" and "surface" meshes, ignore others like "shape_0"
    for obj in bpy.context.scene.objects:
        if obj.name == 'shape_0':
            obj.select_set(True)

    bpy.ops.object.delete()

    # Set output file path
    output_path = os.path.join(output_folder, f"{os.path.splitext(usd_file)[0]}.png")
    bpy.context.scene.render.filepath = output_path

    # Render and save PNG
    bpy.ops.render.render(write_still=True)

    # Deselect all objects
    bpy.ops.object.select_all(action='SELECT')

    # Select only camera, delete everything else
    for obj in bpy.context.scene.objects:
        if obj.name == 'Camera':  # Changed to match case with earlier reference
            obj.select_set(False)

    bpy.ops.object.delete()

print("Rendering complete.")
