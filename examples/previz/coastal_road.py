"""Build a simple previz scene in Blender and render a playblast-style clip (Workbench, solid colours).

A red sports car on a coastal road, cliffs on one side, the sea on the other, filmed from a camera car ahead of it.
Run headless:
  blender -b --factory-startup -P examples/previz/coastal_road.py -- --out ~/.pickups/samples/coastal_road_previz
Then: ffmpeg -framerate 24 -i <out>/f_%04d.png -c:v libx264 -crf 12 -pix_fmt yuv420p coastal_road_previz.mp4
"""

import math
import sys

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
out = argv[argv.index("--out") + 1] if "--out" in argv else "/tmp/coastal_road_previz"

FRAMES, FPS, SPEED = 97, 24, 14.0  # 4 s at 24 fps, car speed in m/s

scene = bpy.context.scene
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o, do_unlink=True)
scene.frame_start, scene.frame_end = 1, FRAMES
scene.render.fps = FPS
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.render.engine = "BLENDER_WORKBENCH"
shading = scene.display.shading
shading.light = "STUDIO"
shading.color_type = "OBJECT"
shading.show_shadows = True
shading.show_cavity = True
scene.display.shadow_focus = 0.6
scene.world = scene.world or bpy.data.worlds.new("World")
scene.world.color = (0.62, 0.72, 0.85)
shading.background_type = "WORLD"
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = f"{out}/f_"


def box(name, loc, size, color, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    o.color = (*color, 1)
    return o


def cyl(name, loc, r, depth, color, rot=(0, 0, 0), verts=24):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=r, depth=depth, location=loc, rotation=rot)
    o = bpy.context.object
    o.name = name
    o.color = (*color, 1)
    return o


# Road along +X, the sea at -Y, cliffs at +Y.
box("road", (60, 0, -0.05), (400, 9, 0.1), (0.22, 0.22, 0.23))
for i in range(80):
    box(f"dash_{i}", (-80 + i * 5, 0, 0.01), (2.0, 0.15, 0.02), (0.9, 0.9, 0.85))
box("verge", (60, 7, -0.06), (400, 5, 0.1), (0.42, 0.45, 0.30))
box("sea", (60, -120, -1.5), (400, 220, 0.1), (0.16, 0.34, 0.5))
box("drop", (60, -6.5, -0.8), (400, 3, 1.6), (0.40, 0.36, 0.30))
for i in range(28):
    h = 8 + 7 * abs(math.sin(i * 1.7))
    box(f"cliff_{i}", (-70 + i * 12, 14 + 3 * math.sin(i), h / 2 - 0.5), (12, 8 + 3 * abs(math.cos(i)), h), (0.55, 0.47, 0.38),
        rot=(0, 0, 0.15 * math.sin(i * 2.3)))
for i in range(40):
    box(f"post_{i}", (-60 + i * 8, -4.9, 0.45), (0.15, 0.15, 0.9), (0.85, 0.85, 0.82))
for i in range(10):
    x = -40 + i * 22
    cyl(f"trunk_{i}", (x, 9.5, 1.5), 0.25, 3.0, (0.35, 0.25, 0.18))
    bpy.ops.mesh.primitive_cone_add(vertices=12, radius1=2.0, depth=5, location=(x, 9.5, 5.0))
    bpy.context.object.color = (0.18, 0.36, 0.2, 1)

# The car: body, cabin, wheels; parented to an empty that moves.
bpy.ops.object.empty_add(location=(0, -1.8, 0))
car = bpy.context.object
car.name = "car"
parts = [box("body", (0, -1.8, 0.75), (4.4, 1.8, 0.7), (0.75, 0.08, 0.06)),
         box("cabin", (-0.3, -1.8, 1.35), (2.0, 1.55, 0.55), (0.70, 0.07, 0.05)),
         box("windscreen", (0.75, -1.8, 1.32), (0.08, 1.45, 0.48), (0.15, 0.18, 0.22), rot=(0, math.radians(-35), 0))]
for dx in (-1.4, 1.4):
    for dy in (-0.85, 0.85):
        parts.append(cyl("wheel", (dx, -1.8 + dy, 0.38), 0.38, 0.28, (0.05, 0.05, 0.05), rot=(math.radians(90), 0, 0)))
for prt in parts:
    prt.parent = car
    prt.matrix_parent_inverse = car.matrix_world.inverted()
car.location = (-10, 0, 0)
car.keyframe_insert("location", frame=1)
car.location = (-10 + SPEED * FRAMES / FPS, 0, 0)
car.keyframe_insert("location", frame=FRAMES)
for fc in car.animation_data.action.fcurves if car.animation_data.action and hasattr(car.animation_data.action, "fcurves") else []:
    for kp in fc.keyframe_points:
        kp.interpolation = "LINEAR"

# Camera: a camera car ahead of the hero car, low, looking back down the road; the car closes in.
# Looking along -X puts the sea on frame left and the cliffs on frame right.
bpy.ops.object.camera_add(location=(70, -1.2, 1.15), rotation=(math.radians(87), 0, math.radians(90)))
cam = bpy.context.object
cam.data.lens = 40
scene.camera = cam
cam.keyframe_insert("location", frame=1)
cam.location = (63, -1.4, 1.2)
cam.keyframe_insert("location", frame=FRAMES)

bpy.ops.object.light_add(type="SUN", rotation=(math.radians(60), 0, math.radians(-40)))
bpy.ops.render.render(animation=True)
print("rendered", out)
