r"""
blender_render_single.py
------------------------
Runs INSIDE Blender (blender --background --python blender_render_single.py).

Usage (called by clip_filter_blender.py):
    blender --background --python blender_render_single.py -- \
        --model C:\path\to\model.glb \
        --out_dir C:\path\to\renders \
        --render_size 224

Outputs:
    <out_dir>\<model_stem>_preview.png   224x224 RGBA, transparent background
"""

import bpy
import sys
import os
import argparse
import logging
import math
from mathutils import Vector

logging.basicConfig(level=logging.INFO, format="[BLENDER][%(levelname)s] %(message)s")
log = logging.getLogger("blender_render_single")


# ──────────────────────────────────────────────
# ARG PARSING  (Blender passes args after "--")
# ──────────────────────────────────────────────
def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser(description="Single-model preview render for CLIP filtering")
    parser.add_argument("--model",        required=True,       help="Path to .glb / .fbx / .obj")
    parser.add_argument("--out_dir",      required=True,       help="Directory to write the PNG into")
    parser.add_argument("--render_size",  type=int, default=224)
    parser.add_argument("--model_height", type=float, default=1.8)
    args = parser.parse_args(argv)
    log.info("Args: %s", vars(args))
    return args


# ──────────────────────────────────────────────
# ENGINE DETECTION  (Blender 3.x vs 4.x)
# ──────────────────────────────────────────────
def get_eevee_engine_name():
    """
    Engine name by Blender version:
      < 4.2 : 'BLENDER_EEVEE'
      4.2.x : 'BLENDER_EEVEE_NEXT'
      5.0+  : 'BLENDER_EEVEE'  (renamed back)
    Detect at runtime by querying the actual enum values Blender exposes.
    """
    import bpy
    allowed = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    if "BLENDER_EEVEE_NEXT" in allowed:
        return "BLENDER_EEVEE_NEXT"
    return "BLENDER_EEVEE"


# ──────────────────────────────────────────────
# SCENE HELPERS
# ──────────────────────────────────────────────
def clear_scene():
    log.info("Clearing scene")
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    # Purge orphan data to avoid memory leaks across renders
    bpy.ops.outliner.orphans_purge(do_local_ids=True, do_linked_ids=False, do_recursive=True)


def create_camera():
    cam_data       = bpy.data.cameras.new("PreviewCam")
    cam_data.type  = 'PERSP'
    cam_data.angle = math.radians(50)
    cam = bpy.data.objects.new("PreviewCam", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    log.info("Camera created (50-deg FOV)")
    return cam


def add_three_point_lights():
    lights = [
        ("Key",  'AREA', (3.0, -4.0, 5.0),  800.0),
        ("Fill", 'AREA', (-4.0, -2.0, 3.0), 300.0),
        ("Rim",  'AREA', (0.0,   5.0, 4.0), 200.0),
    ]
    for name, ltype, loc, energy in lights:
        d          = bpy.data.lights.new(name=name, type=ltype)
        d.energy   = energy
        o          = bpy.data.objects.new(name=name, object_data=d)
        bpy.context.scene.collection.objects.link(o)
        o.location = loc
    log.info("3-point light rig added")


def import_model(path):
    log.info("Importing: %s", path)
    ext    = os.path.splitext(path)[1].lower()
    before = set(bpy.context.scene.objects)

    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        # Blender 3.3+ uses wm.obj_import; older uses import_scene.obj
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError("Unsupported format: " + ext)

    after    = set(bpy.context.scene.objects)
    imported = list(after - before)
    log.info("Imported %d objects", len(imported))
    return imported


def find_meshes(objects):
    meshes = [o for o in objects if o.type == 'MESH']
    log.info("Mesh objects found: %d", len(meshes))
    return meshes


def compute_world_bbox(meshes):
    pts = []
    for m in meshes:
        pts.extend(m.matrix_world @ Vector(c) for c in m.bound_box)
    if not pts:
        raise RuntimeError("Empty bounding box")
    min_v = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
    max_v = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
    return min_v, max_v


def normalize_height(meshes, target):
    min_v, max_v = compute_world_bbox(meshes)
    current      = max_v.z - min_v.z
    if current <= 1e-6:
        log.warning("Degenerate height (%.6f), skipping normalisation", current)
        return
    scale = target / current
    log.info("Height normalise: %.4f -> %.4f  (scale=%.4f)", current, target, scale)
    for m in meshes:
        m.scale = (m.scale.x * scale, m.scale.y * scale, m.scale.z * scale)
        bpy.context.view_layer.objects.active = m
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    # Force Blender to flush matrix_world so compute_world_bbox() sees
    # the updated (post-scale) coordinates when placing the camera.
    bpy.context.view_layer.update()
    log.info("Scene updated after normalisation")


def place_camera_front(cam, meshes):
    min_v, max_v = compute_world_bbox(meshes)
    center       = (min_v + max_v) * 0.5
    height       = max_v.z - min_v.z
    width        = max_v.x - min_v.x
    fov          = cam.data.angle
    extent       = max(height, width) * 0.55   # 0.5 + 10% padding
    dist         = extent / math.tan(fov * 0.5)

    cam.location       = center + Vector((0.0, -dist, 0.0))
    direction          = center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

    log.info(
        "Camera at (%.2f, %.2f, %.2f), dist=%.3f",
        cam.location.x, cam.location.y, cam.location.z, dist
    )


# ──────────────────────────────────────────────
# RENDER
# ──────────────────────────────────────────────
def setup_eevee(res):
    engine = get_eevee_engine_name()
    log.info(
        "Engine: %s  (Blender %s)",
        engine,
        ".".join(str(v) for v in bpy.app.version)
    )
    scene                                        = bpy.context.scene
    scene.render.engine                          = engine
    scene.render.resolution_x                    = res
    scene.render.resolution_y                    = res
    scene.render.resolution_percentage           = 100
    scene.render.film_transparent                = True
    scene.render.image_settings.file_format      = 'PNG'
    scene.render.image_settings.color_mode       = 'RGBA'
    scene.render.image_settings.color_depth      = '8'
    scene.eevee.taa_render_samples               = 16
    log.info("EEVEE set to %dx%d, 16 samples", res, res)


def render_to(path):
    # Must be an absolute path; Blender may append ".png" on some versions
    # if the path already ends in ".png" this is a no-op.
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    log.info("Render written: %s", path)


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def main():
    args    = parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    clear_scene()
    cam = create_camera()
    add_three_point_lights()

    try:
        imported = import_model(args.model)
    except Exception as e:
        log.error("Import failed: %s", e)
        sys.exit(1)

    meshes = find_meshes(imported)
    if not meshes:
        log.error("No mesh objects in %s", args.model)
        sys.exit(1)

    try:
        normalize_height(meshes, args.model_height)
        place_camera_front(cam, meshes)
    except Exception as e:
        log.error("Scene setup error: %s", e)
        sys.exit(1)

    setup_eevee(args.render_size)

    stem     = os.path.splitext(os.path.basename(args.model))[0]
    out_path = os.path.join(out_dir, stem + "_preview.png")

    try:
        render_to(out_path)
    except Exception as e:
        log.error("Render error: %s", e)
        sys.exit(1)

    if not os.path.exists(out_path):
        log.error("Output PNG missing after render: %s", out_path)
        sys.exit(1)

    # Parsed by clip_filter_blender.py — must be exactly this format
    print("RENDER_OK:" + out_path, flush=True)


if __name__ == "__main__":
    main()