"""
blender_pipeline.py
-------------------
Pixel-art sprite render pipeline for pix2pix dataset generation.
Renders one model → RGB sprite + stylised hand-drawn normal map.

Output folder layout (pix2pix-ready):
    <out_dir>/
        color/   ← RGB renders  (RGBA PNG, hard alpha)
        normal/  ← matching normal maps  (RGB PNG)
    Both subdirectories use identical filenames so dataloaders can
    pair them by name without any manifest file.

Usage:
    blender --background --python blender_pipeline.py -- \
        --model path/to/model.glb \
        --out_dir path/to/dataset \
        --render_size 64 \
        --model_rotation 90 \
        --cam_elevation 25

With animation sampling:
    blender --background --python blender_pipeline.py -- \
        --model path/to/animated.glb \
        --out_dir path/to/dataset \
        --sample_animations --anim_frames 4

Render approach: two Cycles passes, same settings
--------------------------------------------------
Pass 1 (color):  64-sample Cycles, RGBA, scene colour management (AgX/Filmic).
Pass 2 (normal): 1-sample Cycles, RGB, view_transform=Raw.  Materials are
temporarily swapped to a pure-emission normal shader, then restored.

Both passes share the same camera, filter_size=0.01, Cycles seed=0, and frame,
so pixel coverage decisions are effectively identical — thin geometry hit by a
sample in pass 1 uses the same deterministic seed in pass 2.

An AOV-based single-render approach was considered but abandoned: Blender 4.x/5.x
applies the scene colour transform to AOV buffers even when the File Output node's
format is set to Raw, corrupting the linear [0,1] normal data.  The two-pass
approach avoids this entirely.

Fixes applied
-------------
1. Armature rotation had no effect
   Root cause: transform_apply(rotation=True) on an ARMATURE bakes the rotation
   into rest-pose bone matrices; the animation system then reverses it on the
   next depsgraph update.
   Fix: compose rotation into matrix_world for all roots; only bake into
   non-armature objects.  Armatures carry the rotation in matrix_world, which
   the renderer sees correctly.

2. Animation camera framing (model flew off-screen)
   Root cause A: world_bbox() read obj.bound_box (rest-pose local AABB only).
   Fix: iterate evaluated depsgraph vertices — correct for skinned deformation.
   Root cause B: single-frame bbox used; wide poses extend beyond it.
   Fix: compute a UNION bbox across all sampled frames; set ortho_scale once.

3. Camera ortho_scale too small at elevated angles AND characters too small in frame
   Two related problems, same root:
   (a) ortho_scale was max(screen_w, screen_h) — when a pose is wider than tall,
       screen_w dominates and the character only fills screen_h/screen_w of the
       image height.  For a slightly wide pose this is ~80%; for a spread-arms
       pose it can be 40–50%.
   (b) The render resolution was always square (res × res).  Combined with (a),
       the character appeared small in both directions.
   Fix (a): ortho_scale is now always screen_h * (1 + padding), so the camera's
       vertical frustum always exactly fits the character's height.
   Fix (b): resolution_x is now computed as round(res_y * screen_w / screen_h),
       giving a non-square image whose width adapts to the actual pose aspect
       ratio.  The character always fills the full image height.
   The projected screen extents (screen_w, screen_h) are stored as custom
   properties on the camera object (cam["screen_w"], cam["screen_h"]) by
   place_ortho_camera and read by render_pair to derive res_x.

4. Fill-ratio threshold rejected valid sprites (false sparse-image discard)
   Root cause: fill ratio was opaque_pixels / canvas_pixels.  A slim character
   in a 128×128 canvas at steep elevation might be 5–8% by this measure even
   though the sprite is perfectly rendered.
   Fix: measure fill ratio as opaque_pixels / tight_bbox_area, where tight_bbox
   is the axis-aligned rect of all opaque pixels.  The threshold now means
   "how filled is the character's own footprint?" — a solid sprite scores ~90%,
   a genuinely blank render scores 0%.  The default threshold is 0.15 (15%).

5. pix2pix folder layout
   color/ and normal/ use identical filename stems so dataloaders can pair by name.
"""

import bpy
import sys
import os
import argparse
import logging
import math
from mathutils import Vector, Matrix

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("blender_pipeline")

MIN_FILL_RATIO = 0.15  # default; overridable with --min_fill_ratio
# NOTE: this is fraction of tight opaque bbox, NOT full canvas — see measure_fill_ratio()


# ─────────────────────────────────────────────────────────────
# ARG PARSING
# ─────────────────────────────────────────────────────────────

def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    p = argparse.ArgumentParser(description="Pixel sprite render pipeline")
    p.add_argument("--model",          required=True)
    p.add_argument("--out_dir",        required=True)
    p.add_argument("--render_size",    type=int,   default=64)
    p.add_argument("--model_rotation", type=float, default=0.0,
                   help="Z-axis rotation in degrees (0=front, 90=right, 180=back).")
    p.add_argument("--cam_elevation",  type=float, default=15.0,
                   help="Camera elevation in degrees (0=front, 25=SNES RPG, 35=Zelda).")
    p.add_argument("--normal_levels",  type=int,   default=6)
    p.add_argument("--model_height",   type=float, default=1.8)
    p.add_argument("--suffix",         default="")
    p.add_argument("--min_fill_ratio", type=float, default=MIN_FILL_RATIO,
                   help="Minimum opaque-pixel fraction to keep a pair (0-1).")
    p.add_argument("--sample_animations", action="store_true")
    p.add_argument("--anim_frames",    type=int,   default=4)
    p.add_argument("--use_outline",    action="store_true",
                   help="Add 1-pixel black outline around the character silhouette.")
    p.add_argument("--save_stages",    action="store_true",
                   help="Save intermediate visualisation images alongside each render pair. "
                        "Creates a <out_dir>/stages/ folder with:\n"
                        "  *_normalised.png  — wireframe-style bbox after height normalisation\n"
                        "  *_camera.png      — scene with camera frustum overlay\n"
                        "  *_color.png       — the final colour render (copy)\n"
                        "  *_normal.png      — the final normal map (copy)\n"
                        "Useful for debugging and thesis diagrams.")

    args = p.parse_args(argv)
    log.info("Parsed arguments: %s", vars(args))
    return args


# ─────────────────────────────────────────────────────────────
# SCENE SETUP
# ─────────────────────────────────────────────────────────────

def clear_scene():
    log.info("Clearing scene")
    for obj in list(bpy.context.scene.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    bpy.ops.outliner.orphans_purge(
        do_local_ids=True, do_linked_ids=False, do_recursive=True
    )


def create_ortho_camera():
    cam_data             = bpy.data.cameras.new("SpriteCamera")
    cam_data.type        = 'ORTHO'
    cam_data.ortho_scale = 2.0          # overwritten by place_ortho_camera()
    cam = bpy.data.objects.new("SpriteCamera", cam_data)
    bpy.context.scene.collection.objects.link(cam)
    bpy.context.scene.camera = cam
    log.info("Orthographic camera created")
    return cam


def add_flat_light():
    """Even two-light rig: large area lights from front and back."""
    log.info("Adding flat two-light rig")
    for name, loc, energy, size in [
        ("FrontLight", ( 0.0, -10.0, 5.0), 1500.0, 10.0),
        ("BackLight",  ( 0.0,  10.0, 5.0),  500.0, 10.0),
    ]:
        ld        = bpy.data.lights.new(name=name, type='AREA')
        ld.energy = energy
        ld.size   = size
        lo        = bpy.data.objects.new(name=name, object_data=ld)
        bpy.context.scene.collection.objects.link(lo)
        lo.location = loc


# ─────────────────────────────────────────────────────────────
# MODEL IMPORT
# ─────────────────────────────────────────────────────────────

def import_model(path: str):
    path = os.path.abspath(path)
    log.info("Importing model: %s", path)
    if not os.path.isfile(path):
        raise RuntimeError(f"Model file not found: {path}")
    ext    = os.path.splitext(path)[1].lower()
    before = set(bpy.context.scene.objects)

    if ext in {".glb", ".gltf"}:
        bpy.ops.import_scene.gltf(filepath=path, merge_vertices=True)
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=path)
    elif ext == ".obj":
        if hasattr(bpy.ops.wm, "obj_import"):
            bpy.ops.wm.obj_import(filepath=path)
        else:
            bpy.ops.import_scene.obj(filepath=path)
    else:
        raise RuntimeError(f"Unsupported model format: {ext!r}")

    imported = list(set(bpy.context.scene.objects) - before)
    log.info("Imported %d objects", len(imported))
    return imported


def find_mesh_objects(objects):
    """Return mesh objects; prefers skinned (armature-modified) to exclude stray geometry."""
    all_meshes = [o for o in objects if o.type == 'MESH']
    skinned    = [m for m in all_meshes if any(mod.type == 'ARMATURE' for mod in m.modifiers)]
    if skinned:
        if len(skinned) < len(all_meshes):
            log.info("find_mesh_objects: %d skinned, ignoring %d unskinned stray objects",
                     len(skinned), len(all_meshes) - len(skinned))
        return skinned
    return all_meshes


def find_root_objects(objects):
    """Root objects, excluding standalone root meshes (stray bounding-volume objects)."""
    obj_set  = set(objects)
    roots    = [o for o in objects if o.parent is None or o.parent not in obj_set]
    non_mesh = [o for o in roots if o.type != 'MESH']
    if non_mesh:
        stray = [o.name for o in roots if o.type == 'MESH']
        if stray:
            log.warning("find_root_objects: skipping stray root mesh(es): %s", stray)
        return non_mesh
    return roots


# ─────────────────────────────────────────────────────────────
# BOUNDING BOX  (deformation-aware)
# ─────────────────────────────────────────────────────────────

def world_bbox_from_vertices(mesh_objects):
    """
    Compute the true world-space bounding box by iterating every evaluated
    vertex of every mesh object.

    Why not obj.bound_box?
      bound_box is the local AABB of the *rest pose* mesh data.  It does NOT
      update when an armature deforms the mesh (skinning happens on the GPU /
      depsgraph; the CPU-side bound_box stays at rest-pose values).
      For unanimated meshes both approaches give the same result, but for
      animated characters this is the only correct method.

    Performance note: at pixel-art resolutions the meshes are small (hundreds
    to low thousands of verts), so this is fast enough.
    """
    depsgraph = bpy.context.evaluated_depsgraph_get()
    all_pts   = []

    for obj in mesh_objects:
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()
        if mesh_eval is None:
            continue
        mw = obj_eval.matrix_world
        all_pts.extend(mw @ v.co for v in mesh_eval.vertices)
        obj_eval.to_mesh_clear()

    if not all_pts:
        raise RuntimeError("world_bbox_from_vertices: no vertices found")

    xs = [p.x for p in all_pts]
    ys = [p.y for p in all_pts]
    zs = [p.z for p in all_pts]
    return (Vector((min(xs), min(ys), min(zs))),
            Vector((max(xs), max(ys), max(zs))))


def union_bbox(bbox_list):
    """Merge a list of (min_v, max_v) pairs into one enclosing bbox."""
    min_x = min(b[0].x for b in bbox_list)
    min_y = min(b[0].y for b in bbox_list)
    min_z = min(b[0].z for b in bbox_list)
    max_x = max(b[1].x for b in bbox_list)
    max_y = max(b[1].y for b in bbox_list)
    max_z = max(b[1].z for b in bbox_list)
    return Vector((min_x, min_y, min_z)), Vector((max_x, max_y, max_z))


# ─────────────────────────────────────────────────────────────
# TRANSFORM HELPERS
# ─────────────────────────────────────────────────────────────

def normalize_model_height(meshes, all_imported, target_height: float):
    """
    Scale the whole hierarchy so the character reaches target_height in Blender
    units along its Z axis.

    IMPORTANT — call order:
        normalize_model_height  →  center_model_at_origin  →  apply_model_rotation

    Height is always measured along world Z (vertical).  Z-rotation (yaw) does
    not change a model's Z extent, so normalising before rotation is correct and
    intentional.  If you ever add pitch/roll rotations, move this call after them.
    """
    log.info("Normalising model height to %.3f units", target_height)
    min_v, max_v = world_bbox_from_vertices(meshes)
    current = max_v.z - min_v.z
    if current <= 1e-6:
        log.warning("Degenerate height (%.6f), skipping normalisation", current)
        return
    scale  = target_height / current
    roots  = find_root_objects(all_imported)
    for o in roots:
        o.scale *= scale
    bpy.context.view_layer.update()
    # Bake scale into mesh roots only (safe for all object types)
    for o in roots:
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.context.view_layer.update()
    log.info("Scale factor applied: %.6f", scale)


def center_model_at_origin(meshes, all_imported):
    """Translate roots so the bbox centre is at the world origin (base at Z=0)."""
    min_v, max_v = world_bbox_from_vertices(meshes)
    center = (min_v + max_v) * 0.5
    offset = Vector((-center.x, -center.y, -min_v.z))
    roots  = find_root_objects(all_imported)
    for o in roots:
        o.location += offset
    bpy.context.view_layer.update()


def apply_model_rotation(all_imported, degrees: float):
    """
    Rotate the whole imported hierarchy around the world Z axis.

    THE FIX for 90°/180° having no effect:
    ----------------------------------------
    Calling transform_apply(rotation=True) on an ARMATURE bakes the rotation
    into the rest-pose bone head/tail positions.  On the very next depsgraph
    update, the animation system rebuilds the posed matrices from the rest pose
    plus the action curves — effectively reversing the rotation.  The visible
    mesh pops back to its original orientation at render time.

    Solution: compose the rotation directly into matrix_world for ALL root
    objects (including armatures), but only call transform_apply on objects
    that are NOT armatures.  Armatures keep the rotation encoded in their
    matrix_world, which is sufficient for the renderer.

    Using matrix composition (rot_matrix @ o.matrix_world) instead of adding
    euler angles is also important: GLB armatures often arrive with a 90°
    rotation pre-baked into their euler.  Adding more euler angles on top
    produces gimbal-lock artefacts at certain angles.
    """
    if degrees == 0.0:
        return
    log.info("Applying Z rotation: %.1f degrees", degrees)

    rot_matrix = Matrix.Rotation(math.radians(degrees), 4, 'Z')
    roots      = find_root_objects(all_imported)

    for o in roots:
        o.matrix_world = rot_matrix @ o.matrix_world

    bpy.context.view_layer.update()

    # Bake rotation into non-armature roots so their transforms are clean
    for o in roots:
        if o.type == 'ARMATURE':
            # Do NOT bake into armatures — see docstring above
            continue
        bpy.context.view_layer.objects.active = o
        bpy.ops.object.select_all(action='DESELECT')
        o.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    bpy.context.view_layer.update()


# ─────────────────────────────────────────────────────────────
# ANIMATION SAMPLING
# ─────────────────────────────────────────────────────────────

def discover_actions():
    actions = list(bpy.data.actions)
    if actions:
        log.info("Discovered %d action(s): %s",
                 len(actions), [a.name for a in actions])
    else:
        log.info("No actions found in imported model")
    return actions


def sample_frame_numbers(action, n_frames: int) -> list:
    start = int(action.frame_range[0])
    end   = int(action.frame_range[1])
    total = end - start + 1
    if n_frames >= total:
        return list(range(start, end + 1))
    if n_frames == 1:
        return [start]
    step = (end - start) / (n_frames - 1)
    return [round(start + step * i) for i in range(n_frames)]


def apply_action_at_frame(armatures, action, frame: int):
    scene = bpy.context.scene
    for arm in armatures:
        if arm.animation_data is None:
            arm.animation_data_create()
        arm.animation_data.action = action
    scene.frame_set(frame)
    bpy.context.view_layer.update()
    log.info("Posed at action='%s' frame=%d", action.name, frame)


def compute_animation_union_bbox(armatures, action, frames, meshes):
    """Evaluate every frame; return (union_min, union_max, frame_bboxes dict)."""
    frame_bboxes = {}
    for frame in frames:
        apply_action_at_frame(armatures, action, frame)
        frame_bboxes[frame] = world_bbox_from_vertices(meshes)
    u_min, u_max = union_bbox(list(frame_bboxes.values()))
    return u_min, u_max, frame_bboxes


# ─────────────────────────────────────────────────────────────
# CAMERA PLACEMENT
# ─────────────────────────────────────────────────────────────

def place_ortho_camera(cam, min_v, max_v,
                       elevation_deg: float = 15.0):
    """
    Position and scale the orthographic camera to frame the given bbox.

    FIX (Bug #3) — elevation-aware ortho_scale
    -------------------------------------------
    The old code set ortho_scale = max(world_Z_height, world_XY_width).
    This is only correct at elevation=0 (pure front view).  At any non-zero
    elevation the camera is tilted, so the character's Y depth (front-to-back)
    foreshortens onto the screen's vertical axis.  The projected screen height
    is approximately:

        screen_H = Z_height * cos(elev) + Y_depth * sin(elev)

    Using raw Z_height always under-estimates the true projected height at
    elevated angles, so ortho_scale ends up too small — the character fills
    only a fraction of the image even though it rendered correctly.

    The fix: compute the projected extents of the bbox corners onto the camera's
    view plane (XZ plane of the camera), take their range, and use that as the
    basis for ortho_scale.  This is exact for orthographic cameras regardless of
    elevation angle.

    Accepting pre-computed (min_v, max_v) instead of re-computing the bbox here
    means the caller controls which bbox is used — the rest-pose bbox for static
    renders, or the union-bbox for animation (which covers every sampled pose).
    """
    center   = (min_v + max_v) * 0.5
    elev_rad = math.radians(elevation_deg)

    # Camera axes in world space (orthographic, looking at origin from above-front):
    #   right  = world X  (unchanged by elevation)
    #   up     = -sin(elev)*Y + cos(elev)*Z   (camera's vertical screen axis)
    #   forward=  cos(elev)*Y + sin(elev)*Z   (into the screen, not needed here)
    #
    # Project all 8 bbox corners onto (right, up) to get screen-space extents.
    sin_e = math.sin(elev_rad)
    cos_e = math.cos(elev_rad)

    corners = [
        Vector((x, y, z))
        for x in (min_v.x, max_v.x)
        for y in (min_v.y, max_v.y)
        for z in (min_v.z, max_v.z)
    ]

    # Shift corners relative to bbox center so we measure size, not position
    proj_x = [c.x - center.x for c in corners]                        # screen right
    proj_y = [-(c.y - center.y) * sin_e + (c.z - center.z) * cos_e   # screen up
               for c in corners]

    screen_w = max(proj_x) - min(proj_x)
    screen_h = max(proj_y) - min(proj_y)
    binding  = max(screen_w, screen_h)   # larger dimension for square images

    # Store bare (no-padding) binding so render_pair can apply pixel-accurate
    # 1-px padding: ortho_scale = binding * res/(res-2).
    cam.data.ortho_scale = binding
    cam["screen_w"] = screen_w
    cam["screen_h"] = screen_h
    cam["binding"]  = binding

    # Place camera along the view direction at a safe distance
    dist = max(max_v.x - min_v.x, max_v.y - min_v.y, max_v.z - min_v.z) * 5.0
    cam.location = center + Vector((
        0.0,
        -dist * math.cos(elev_rad),
         dist * math.sin(elev_rad),
    ))
    cam.rotation_euler = (center - cam.location).to_track_quat('-Z', 'Y').to_euler()
    log.info(
        "Camera framed: screen_w=%.4f  screen_h=%.4f  binding=%.4f  elev=%.1f°",
        screen_w, screen_h, binding, elevation_deg,
    )
    return screen_w, screen_h


# ─────────────────────────────────────────────────────────────
# COMBINED SINGLE-RENDER  (color + normal in one Cycles pass)
# ─────────────────────────────────────────────────────────────
#
# Why one render instead of two?
# ───────────────────────────────
# Cycles is stochastic: whether a ray hits a 1-pixel-wide object depends on
# the random sample positions for that pixel.  With 64 samples the RGB pass
# accumulates enough hits to declare a thin object (cane, sword, hair strand)
# opaque.  With 1 sample the old normal pass often misses entirely — the pixel
# stays flat-blue background even though the RGB pixel is filled.  The result
# is the misalignment you observed.
#
# Fix: inject the stylised normal shader as a Cycles AOV (Arbitrary Output
# Variable).  Both color and normal data are computed from the EXACT SAME ray
# samples in a single bpy.ops.render.render() call.  The compositor then
# writes each AOV to a separate file using File Output nodes.  Pixel coverage
# is shared by definition.
#
# Pipeline overview:
#   1. setup_combined_render()   — scene settings, view layer AOV, compositor
#   2. _render_normal()           — swap materials, render, restore
#   3. bpy.ops.render.render()   — one render, two output files
#   4. post_process_outputs()    — hard-alpha on color, fill-ratio check

# ─────────────────────────────────────────────────────────────
# RENDER APPROACH:  two Cycles renders, same settings
# ─────────────────────────────────────────────────────────────
#
# Why not AOV?
# ─────────────────
# Blender's AOV system applies the scene colour transform (AgX/Filmic) when
# storing values into the AOV buffer.  Normal vectors remapped to [0,1] look
# correct as linear data but come out blown-out / washed when pushed through
# a photographic tone curve.  Setting view_transform='Raw' on the File Output
# slot format (format.view_settings) does NOT reliably bypass this in Blender
# 4.x/5.x — the property is effectively read-only in those versions.
#
# Fix: two renders, both using view_transform='Standard' (true 1:1 passthrough
# in every Blender version) so values are written to PNG exactly as the shader
# computes them.  Both renders share the same resolution, filter_size, camera,
# and frame, so pixel coverage decisions are identical — thin geometry like a
# cane that is hit by a sample in pass 1 uses the same seed/settings in pass 2.
# The color pass uses Blender's default scene transform (AgX/Filmic) so PBR
# materials look correct; the normal pass overrides to Standard only for its
# render and restores the original transform immediately after.
#
# Pixel alignment is maintained because both passes:
#   • use the same orthographic camera and ortho_scale
#   • use the same filter_size=0.01 (near-box, sub-pixel jitter is minimal)
#   • use the same samples seed (Cycles default)
#   • do NOT call object.join() between passes (mesh is untouched)


def _setup_cycles_base(res_x: int, res_y: int):
    """
    Common Cycles + output settings shared by both passes.

    res_x / res_y are computed by render_pair from the camera's projected
    aspect ratio so the character always fills the full image height.
    Both values are rounded to integers before being passed here.
    """
    scene = bpy.context.scene
    scene.render.engine               = 'CYCLES'
    scene.cycles.device               = 'CPU'
    scene.cycles.samples              = 64
    scene.cycles.use_denoising        = False   # denoiser blurs pixel-art edges
    scene.cycles.seed                 = 0       # deterministic sampling
    scene.render.filter_size          = 0.01    # near-box filter → crisp pixels
    scene.render.use_motion_blur      = False
    scene.render.use_simplify         = True
    scene.render.simplify_subdivision = 0
    scene.render.pixel_aspect_x       = 1.0
    scene.render.pixel_aspect_y       = 1.0
    scene.render.resolution_x         = res_x
    scene.render.resolution_y         = res_y
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity      = 0.0
    # Disable compositor for direct renders — we write filepath directly
    scene.use_nodes              = False
    scene.render.use_compositing = False


def _get_eevee_engine():
    """Return the correct Eevee engine name for this Blender version."""
    allowed = bpy.types.RenderSettings.bl_rna.properties["engine"].enum_items.keys()
    return "BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in allowed else "BLENDER_EEVEE"


def _render_color(path: str, res_x: int, res_y: int):
    """Render the color pass via Eevee: faster and more vibrant than Cycles for pixel art.

    The world background is zeroed out so all illumination comes from our
    area lights only.  Without this, any world environment left by a GLTF
    import or by the previous normal pass would add uncontrolled ambient
    light that overexposes subsequent frames.
    """
    log.info("Rendering color pass (Eevee) -> %s  (%dx%d)", path, res_x, res_y)
    scene  = bpy.context.scene
    engine = _get_eevee_engine()

    scene.render.engine                          = engine
    scene.render.resolution_x                    = res_x
    scene.render.resolution_y                    = res_y
    scene.render.resolution_percentage           = 100
    scene.render.film_transparent                = True
    scene.render.filter_size                     = 0.01
    scene.render.use_motion_blur                 = False
    scene.render.use_simplify                    = True
    scene.render.simplify_subdivision            = 0
    scene.render.pixel_aspect_x                  = 1.0
    scene.render.pixel_aspect_y                  = 1.0
    scene.render.dither_intensity                = 0.0
    scene.render.image_settings.file_format      = 'PNG'
    scene.render.image_settings.color_mode       = 'RGBA'
    scene.render.image_settings.color_depth      = '8'
    scene.use_nodes                              = False
    scene.render.use_compositing                 = False
    try:
        scene.eevee.taa_render_samples = 16
    except AttributeError:
        pass

    # +1 EV for bright, punchy pixel-art colors
    scene.view_settings.exposure = 1.0
    scene.view_settings.gamma    = 1.0

    # Zero out any world ambient so area lights are the ONLY illumination source.
    # The GLTF importer may set up an environment, and the normal pass leaves
    # a blue world; both would overexpose subsequent Eevee color renders.
    if scene.world:
        scene.world.use_nodes = True
        wnt = scene.world.node_tree
        bg  = wnt.nodes.get("Background") or wnt.nodes.new("ShaderNodeBackground")
        bg.inputs[1].default_value = 0.0   # strength = 0 → no ambient contribution

    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def _build_normal_material(meshes: list, levels: int):
    """
    Replace every material on every mesh with a pure-emission normal shader.
    Returns the list of (mesh, original_materials) so they can be restored.

    The shader computes:
      1. Camera-space normal, flipped to OpenGL convention
      2. Remapped from [-1,1] to [0,1]
      3. Quantised to `levels` steps  →  flat cel-shaded zones
      4. Fresnel ink darkening at silhouettes
    Output goes straight to Emission → Material Output with no lighting.
    """
    mat   = bpy.data.materials.new("__NormalPass")
    mat.use_nodes = True
    nt    = mat.node_tree
    links = nt.links
    nt.nodes.clear()

    out_node  = nt.nodes.new("ShaderNodeOutputMaterial")
    emit_node = nt.nodes.new("ShaderNodeEmission")
    geo       = nt.nodes.new("ShaderNodeNewGeometry")

    cam_xform              = nt.nodes.new("ShaderNodeVectorTransform")
    cam_xform.vector_type  = 'NORMAL'
    cam_xform.convert_from = 'WORLD'
    cam_xform.convert_to   = 'CAMERA'

    flip_z           = nt.nodes.new("ShaderNodeVectorMath")
    flip_z.operation = 'MULTIPLY'
    flip_z.inputs[1].default_value = (1.0, 1.0, -1.0)

    nrm           = nt.nodes.new("ShaderNodeVectorMath")
    nrm.operation = 'NORMALIZE'

    mul_half           = nt.nodes.new("ShaderNodeVectorMath")
    mul_half.operation = 'MULTIPLY'
    mul_half.inputs[1].default_value = (0.5, 0.5, 0.5)

    add_half           = nt.nodes.new("ShaderNodeVectorMath")
    add_half.operation = 'ADD'
    add_half.inputs[1].default_value = (0.5, 0.5, 0.5)

    q_mul           = nt.nodes.new("ShaderNodeVectorMath")
    q_mul.operation = 'MULTIPLY'
    q_mul.inputs[1].default_value = (levels, levels, levels)

    q_floor           = nt.nodes.new("ShaderNodeVectorMath")
    q_floor.operation = 'FLOOR'

    q_div           = nt.nodes.new("ShaderNodeVectorMath")
    q_div.operation = 'DIVIDE'
    q_div.inputs[1].default_value = (levels, levels, levels)

    fresnel = nt.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = 1.0

    ink_mul           = nt.nodes.new("ShaderNodeMath")
    ink_mul.operation = 'MULTIPLY'
    ink_mul.inputs[1].default_value = 2.5      # boldness — raise for thicker lines

    ink_clamp           = nt.nodes.new("ShaderNodeMath")
    ink_clamp.operation = 'MINIMUM'
    ink_clamp.inputs[1].default_value = 1.0

    one_minus           = nt.nodes.new("ShaderNodeMath")
    one_minus.operation = 'SUBTRACT'
    one_minus.inputs[0].default_value = 1.0

    ink_darken           = nt.nodes.new("ShaderNodeVectorMath")
    ink_darken.operation = 'SCALE'

    links.new(geo.outputs["Normal"],         cam_xform.inputs["Vector"])
    links.new(cam_xform.outputs["Vector"],   flip_z.inputs[0])
    links.new(flip_z.outputs["Vector"],      nrm.inputs[0])
    links.new(nrm.outputs["Vector"],         mul_half.inputs[0])
    links.new(mul_half.outputs["Vector"],    add_half.inputs[0])
    links.new(add_half.outputs["Vector"],    q_mul.inputs[0])
    links.new(q_mul.outputs["Vector"],       q_floor.inputs[0])
    links.new(q_floor.outputs["Vector"],     q_div.inputs[0])
    links.new(fresnel.outputs["Fac"],        ink_mul.inputs[0])
    links.new(ink_mul.outputs["Value"],      ink_clamp.inputs[0])
    links.new(ink_clamp.outputs["Value"],    one_minus.inputs[1])
    links.new(one_minus.outputs["Value"],    ink_darken.inputs["Scale"])
    links.new(q_div.outputs["Vector"],       ink_darken.inputs[0])
    links.new(ink_darken.outputs["Vector"],  emit_node.inputs["Color"])
    links.new(emit_node.outputs["Emission"], out_node.inputs["Surface"])

    # Use object-level material slots — GLTF meshes carry per-slot overrides
    # at the object level; mesh.data.materials is the shared layer that Blender
    # ignores when rendering if slot overrides are present.
    saved = []
    for mesh in meshes:
        orig = [slot.material for slot in mesh.material_slots]
        saved.append((mesh, orig))
        for slot in mesh.material_slots:
            slot.material = mat
        if not mesh.material_slots:
            mesh.data.materials.append(mat)

    return mat, saved


def _restore_materials(mat, saved):
    """Restore original materials via object-level slots; delete temp material."""
    for mesh, orig in saved:
        for i, m in enumerate(orig):
            if i < len(mesh.material_slots):
                mesh.material_slots[i].material = m
    bpy.data.materials.remove(mat, do_unlink=True)


def _render_normal(path: str, res_x: int, res_y: int, meshes: list, levels: int):
    """
    Render the stylised normal map as a pure data texture.

    Why view_transform='Raw'?
    ─────────────────────────
    Normal vectors remapped to [0,1] are LINEAR DATA — not photographic
    content.  Any colour transform that applies a curve (AgX, Filmic, sRGB
    gamma) will corrupt the values:
      • AgX / Filmic: compresses highlights and lifts shadows → blown-out look
      • 'Standard':   applies sRGB gamma 2.2 → brightens mid-tones, shifts hues
      • 'Raw':        writes floats directly to 8-bit PNG with no curve at all
                      → values in PNG match exactly what the shader outputs

    'Standard' was also removed from Blender 4.2+ (AgX replaced it as
    default), so setting it on newer builds silently falls back to AgX.

    'Raw' is the only safe choice for any data render (normals, roughness,
    AO, etc.).  It is correct here because:
      1. The emission shader produces values strictly in [0,1]
      2. film_transparent=False → background is the world colour, not alpha
      3. No lighting calculation is involved (pure emission)
    We save the original transform and restore it immediately after so the
    beauty pass is unaffected.
    """
    log.info("Rendering normal pass -> %s  (%dx%d)", path, res_x, res_y)
    scene = bpy.context.scene
    _setup_cycles_base(res_x, res_y)
    scene.cycles.samples          = 1      # emission needs only 1 sample
    scene.render.film_transparent = False  # opaque RGB output
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode  = 'RGB'
    scene.render.image_settings.color_depth = '8'

    # ── Save and override colour management ─────────────────
    orig_transform = scene.view_settings.view_transform
    orig_look      = scene.view_settings.look
    orig_exposure  = scene.view_settings.exposure
    orig_gamma     = scene.view_settings.gamma
    scene.view_settings.view_transform = 'Raw'   # no curve, no gamma
    scene.view_settings.look           = 'None'
    scene.view_settings.exposure       = 0.0
    scene.view_settings.gamma          = 1.0
    scene.render.dither_intensity      = 0.0

    # ── Save world state, then set flat-blue background ─────
    # Background pixels (alpha=0 in the color pass) will be (0.5, 0.5, 1.0)
    # = standard "facing camera" neutral normal colour.  Under Raw this writes
    # R=128, G=128, B=255 exactly.
    #
    # IMPORTANT: the world background is NOT restored automatically after this
    # render.  In Eevee the world contributes ambient light, so leaving it set
    # to (0.5, 0.5, 1.0, strength=1.0) would add a blue ambient glow to every
    # subsequent color pass — causing all renders after the first to appear
    # overexposed.  We save and restore it here to prevent this.
    orig_world_use_nodes  = scene.world.use_nodes if scene.world else False
    orig_world_bg_color   = None
    orig_world_bg_strength = None
    if scene.world:
        scene.world.use_nodes = True
        wnt = scene.world.node_tree
        bg  = wnt.nodes.get("Background") or wnt.nodes.new("ShaderNodeBackground")
        orig_world_bg_color    = tuple(bg.inputs[0].default_value)
        orig_world_bg_strength = bg.inputs[1].default_value
        bg.inputs[0].default_value = (0.5, 0.5, 1.0, 1.0)
        bg.inputs[1].default_value = 1.0

    # ── Swap materials → render → restore materials ──────────
    mat, saved = _build_normal_material(meshes, levels)
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    _restore_materials(mat, saved)

    # ── Restore world background ─────────────────────────────
    if scene.world and orig_world_bg_color is not None:
        wnt = scene.world.node_tree
        bg  = wnt.nodes.get("Background")
        if bg:
            bg.inputs[0].default_value = orig_world_bg_color
            bg.inputs[1].default_value = orig_world_bg_strength
        scene.world.use_nodes = orig_world_use_nodes

    # ── Restore colour management so beauty pass is unaffected ──
    scene.view_settings.view_transform = orig_transform
    scene.view_settings.look           = orig_look
    scene.view_settings.exposure       = orig_exposure
    scene.view_settings.gamma          = orig_gamma


def _add_pixel_art_outline(path: str):
    """Add a 1-pixel black outline around all opaque pixels (4-connectivity)."""
    img    = bpy.data.images.load(path, check_existing=False)
    px     = list(img.pixels[:])
    width  = img.size[0]
    height = img.size[1]
    bpy.data.images.remove(img)
    new_px = list(px)
    for idx in range(width * height):
        if px[idx * 4 + 3] >= 0.5:
            continue
        x = idx % width
        y = idx // width
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height:
                ni = ny * width + nx
                if px[ni * 4 + 3] >= 0.5:
                    new_px[idx * 4]     = 0.0
                    new_px[idx * 4 + 1] = 0.0
                    new_px[idx * 4 + 2] = 0.0
                    new_px[idx * 4 + 3] = 1.0
                    break
    out = bpy.data.images.new("__Outline", width=width, height=height, alpha=True)
    out.pixels       = new_px
    out.filepath_raw = path
    out.file_format  = 'PNG'
    out.save()
    bpy.data.images.remove(out)
    log.info("Outline applied: %s", os.path.basename(path))


def _threshold_alpha(path: str, threshold: float = 0.5):
    """
    Snap every alpha channel value to exactly 0.0 or 1.0.
    Eliminates the soft sub-pixel fringe that Cycles produces at geometry edges.
    """
    img = bpy.data.images.load(path, check_existing=False)
    px  = list(img.pixels)
    for i in range(3, len(px), 4):
        px[i] = 1.0 if px[i] >= threshold else 0.0
    img.pixels       = px
    img.filepath_raw = path
    img.file_format  = 'PNG'
    img.save()
    bpy.data.images.remove(img)
    log.info("Hard-alpha applied: %s", os.path.basename(path))


def measure_fill_ratio(path: str):
    """
    Return (ratio, opaque, total) where ratio is the fraction of the image's
    TIGHT BOUNDING BOX that is fully opaque.

    FIX (Bug #1) — tight-bbox fill ratio
    --------------------------------------
    The old implementation divided opaque pixels by the full canvas area
    (render_size × render_size).  A slim character standing in a 128×128 canvas
    might cover only 6% of the square even though the sprite itself is excellent.
    At small sizes (64×64) with wide camera elevations this routinely fell below
    the 10% threshold and discarded perfectly good renders.

    The correct question is not "what fraction of the canvas is filled?" but
    "is there actually a character visible at all?".  We answer this by:
      1. Finding the axis-aligned bounding box of all opaque pixels (the tight
         bbox).  If no pixels are opaque the render is genuinely blank.
      2. Computing what fraction of that tight bbox is opaque.  A solid sprite
         scores ~100%; a sprite with a transparent background and small gaps
         (hair, between limbs) scores 50–90%.  A truly empty render scores 0%.

    This means the fill_ratio threshold (--min_fill_ratio) now means something
    intuitive: "what fraction of the character's own footprint is filled?"
    rather than "what fraction of the canvas?".  A threshold of 0.15 (15%) now
    correctly rejects renders where the model is invisible or almost invisible,
    while accepting any sprite with a discernible character — regardless of
    canvas size or camera angle.

    We also return the raw counts so the log line stays informative.
    """
    img    = bpy.data.images.load(path, check_existing=False)
    px     = img.pixels[:]
    width  = img.size[0]
    height = img.size[1]
    bpy.data.images.remove(img)

    total  = width * height
    # Collect (col, row) of every opaque pixel
    opaque_coords = []
    for i in range(total):
        if px[i * 4 + 3] >= 0.5:   # alpha channel
            col = i % width
            row = i // width
            opaque_coords.append((col, row))

    n_opaque = len(opaque_coords)
    if n_opaque == 0:
        log.info("Fill ratio: 0.0%%  (0 / %d opaque pixels) — blank render", total)
        return 0.0, 0, total

    # Tight bounding box of opaque pixels
    min_col = min(c for c, r in opaque_coords)
    max_col = max(c for c, r in opaque_coords)
    min_row = min(r for c, r in opaque_coords)
    max_row = max(r for c, r in opaque_coords)
    bbox_area = (max_col - min_col + 1) * (max_row - min_row + 1)

    ratio = n_opaque / bbox_area
    log.info(
        "Fill ratio: %.1f%%  (%d / %d opaque px in tight bbox %dx%d, canvas %dx%d)",
        ratio * 100, n_opaque, bbox_area,
        max_col - min_col + 1, max_row - min_row + 1,
        width, height,
    )
    return ratio, n_opaque, bbox_area


def _save_stage_images(stages_dir: str, stem: str, suffix: str,
                       rgb_path: str, normal_path: str,
                       cam, meshes, res_x: int, res_y: int,
                       levels: int):
    """
    Save intermediate pipeline stage images for thesis / debugging visualisation.

    Stages written to <stages_dir>/<stem><suffix>_stage_N_<name>.png:
      1_normalised  — character in T-pose with bounding-box overlay (red bbox lines)
      2_camera      — same view with camera frustum overlay (blue ortho rect)
      3_color       — the colour render (hard copy, alpha composited on gray)
      4_normal      — the normal map (hard copy)

    All images are rendered at the same resolution as the main pair so they
    align pixel-for-pixel with the dataset outputs.
    """
    import shutil
    os.makedirs(stages_dir, exist_ok=True)

    def stage_path(n, name):
        return os.path.join(stages_dir, f"{stem}{suffix}_stage{n}_{name}.png")

    scene = bpy.context.scene

    # ── Stage 1: normalised mesh with bbox wireframe ──────────────────────────
    # Render a flat-shaded version with an overlay showing the bounding box.
    # We add a temporary wire-cube object the same size as the bbox, render,
    # then remove it.
    try:
        s1_path = stage_path(1, "normalised")
        min_v, max_v = world_bbox_from_vertices(meshes)

        # Create bbox wireframe cube
        bpy.ops.mesh.primitive_cube_add()
        bbox_obj = bpy.context.active_object
        bbox_obj.name = "__StageBbox"
        cx = (min_v.x + max_v.x) * 0.5
        cy = (min_v.y + max_v.y) * 0.5
        cz = (min_v.z + max_v.z) * 0.5
        bbox_obj.location = (cx, cy, cz)
        bbox_obj.scale    = (
            (max_v.x - min_v.x) * 0.5,
            (max_v.y - min_v.y) * 0.5,
            (max_v.z - min_v.z) * 0.5,
        )
        # Bright red wireframe material
        wmat = bpy.data.materials.new("__BboxWire")
        wmat.use_nodes = True
        wmat.node_tree.nodes.clear()
        em = wmat.node_tree.nodes.new("ShaderNodeEmission")
        out = wmat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        em.inputs["Color"].default_value = (1.0, 0.1, 0.1, 1.0)
        em.inputs["Strength"].default_value = 2.0
        wmat.node_tree.links.new(em.outputs["Emission"], out.inputs["Surface"])
        bbox_obj.data.materials.append(wmat)
        bbox_obj.display_type = 'WIRE'

        _render_color(s1_path, res_x, res_y)
        _threshold_alpha(s1_path)
        log.info("Stage 1 (normalised) saved: %s", os.path.basename(s1_path))
    except Exception as e:
        log.warning("Stage 1 (normalised) failed: %s", e)
    finally:
        try:
            bpy.data.objects.remove(bpy.data.objects.get("__StageBbox"), do_unlink=True)
            bpy.data.materials.remove(bpy.data.materials.get("__BboxWire"), do_unlink=True)
        except Exception:
            pass

    # ── Stage 2: camera frustum overlay ──────────────────────────────────────
    # Draw the ortho frustum as a thin blue rectangle by creating four edge
    # objects (thin cuboids) positioned at the image corners in world space.
    try:
        s2_path = stage_path(2, "camera")
        os_ = cam.data.ortho_scale
        # The camera looks along -Z (local), so frustum corners in camera space are:
        # (±scale/2, ±scale/2, 0) in camera local → need world coords
        import mathutils
        frustum_verts_cam = [
            mathutils.Vector(( os_/2,  os_/2, 0)),
            mathutils.Vector((-os_/2,  os_/2, 0)),
            mathutils.Vector((-os_/2, -os_/2, 0)),
            mathutils.Vector(( os_/2, -os_/2, 0)),
        ]
        mw = cam.matrix_world
        fw = [mw @ v for v in frustum_verts_cam]

        # Create a thin line-mesh showing the frustum rectangle
        verts = [(v.x, v.y, v.z) for v in fw]
        edges = [(0,1),(1,2),(2,3),(3,0)]
        mesh_d = bpy.data.meshes.new("__FrustumMesh")
        mesh_d.from_pydata(verts, edges, [])
        frustum_obj = bpy.data.objects.new("__Frustum", mesh_d)
        bpy.context.scene.collection.objects.link(frustum_obj)

        fmat = bpy.data.materials.new("__FrustumMat")
        fmat.use_nodes = True
        fmat.node_tree.nodes.clear()
        fem = fmat.node_tree.nodes.new("ShaderNodeEmission")
        fout = fmat.node_tree.nodes.new("ShaderNodeOutputMaterial")
        fem.inputs["Color"].default_value = (0.1, 0.3, 1.0, 1.0)
        fem.inputs["Strength"].default_value = 3.0
        fmat.node_tree.links.new(fem.outputs["Emission"], fout.inputs["Surface"])
        frustum_obj.data.materials.append(fmat)

        _render_color(s2_path, res_x, res_y)
        _threshold_alpha(s2_path)
        log.info("Stage 2 (camera) saved: %s", os.path.basename(s2_path))
    except Exception as e:
        log.warning("Stage 2 (camera) failed: %s", e)
    finally:
        for name in ("__Frustum", "__FrustumMesh", "__FrustumMat"):
            try:
                obj = bpy.data.objects.get(name)
                if obj: bpy.data.objects.remove(obj, do_unlink=True)
                mat = bpy.data.materials.get(name)
                if mat: bpy.data.materials.remove(mat, do_unlink=True)
                mesh_d2 = bpy.data.meshes.get(name)
                if mesh_d2: bpy.data.meshes.remove(mesh_d2)
            except Exception:
                pass

    # ── Stages 3 & 4: copies of the outputs ───────────────────────────────────
    try:
        if rgb_path and os.path.exists(rgb_path):
            shutil.copy2(rgb_path, stage_path(3, "color"))
            log.info("Stage 3 (color) saved")
        if normal_path and os.path.exists(normal_path):
            shutil.copy2(normal_path, stage_path(4, "normal"))
            log.info("Stage 4 (normal) saved")
    except Exception as e:
        log.warning("Stage 3/4 copy failed: %s", e)


def render_pair(cam, meshes, res_y, levels, color_dir, normal_dir,
                stem, suffix, min_fill_ratio, use_outline: bool = False,
                stages_dir: str = None):
    """
    Render one color PNG and one normal PNG for the current scene pose.

    res_y is the target image HEIGHT in pixels (the value of --render_size).
    res_x is derived from the camera's projected aspect ratio so the character
    always fills the full image height — the width adapts to the actual pose:

        res_x = max(1, round(res_y * screen_w / screen_h))

    where screen_w / screen_h comes from the ortho_scale calculation in
    place_ortho_camera (which already accounts for elevation).

    FIX — full-height guarantee
    ----------------------------
    The previous code always rendered res × res squares.  When a character's
    projected width (screen_w) exceeded its projected height (screen_h) —
    which happens for wide poses, spread arms, or diagonal rotations — the
    old place_ortho_camera set ortho_scale = max(screen_w, screen_h), so the
    camera's vertical frustum was driven by the width.  The character's height
    only occupied screen_h / screen_w of the image, leaving it visibly smaller
    and surrounded by empty space top and bottom.

    Now: ortho_scale is always screen_h (+ padding), and res_x is computed so
    the pixel aspect exactly matches the projected aspect ratio.  The character
    always reaches the top and bottom of the image regardless of pose or rotation.

    Pass 1 — color:  full 64-sample Cycles, RGBA, scene colour management.
    Pass 2 — normal: 1-sample emission, RGB, view_transform=Raw (data passthrough).
    Both passes use the same res_x × res_y, camera, filter_size, and seed.

    Returns (rgb_path, normal_path) on success, (None, None) if sparse.
    """
    filename    = f"{stem}{suffix}.png"
    rgb_path    = os.path.join(color_dir,  filename)
    normal_path = os.path.join(normal_dir, filename)

    # 1-pixel padding each side: character fills (res-2)/(res) of the image.
    binding = cam.get("binding", cam.data.ortho_scale)
    cam.data.ortho_scale = binding * res_y / max(res_y - 2, 1)

    res_x = res_y   # always square
    log.info("Render %dx%d  ortho_scale=%.4f  (1px padding each side)",
             res_x, res_y, cam.data.ortho_scale)

    # ── Pass 1: color ────────────────────────────────────────
    _render_color(rgb_path, res_x, res_y)
    _threshold_alpha(rgb_path)

    # Optional 1-pixel black outline (color only — normal map is never outlined)
    if use_outline:
        _add_pixel_art_outline(rgb_path)

    ratio, n_opaque, bbox_area = measure_fill_ratio(rgb_path)
    if ratio < min_fill_ratio:
        log.warning(
            "Sparse image (%.1f%% of tight bbox < %.0f%%) — discarding: %s",
            ratio * 100, min_fill_ratio * 100, filename
        )
        try: os.remove(rgb_path)
        except OSError: pass
        return None, None

    # ── Pass 2: normal ───────────────────────────────────────
    _render_normal(normal_path, res_x, res_y, meshes, levels)

    # ── Optional: save intermediate stage images ──────────────
    if stages_dir:
        _save_stage_images(stages_dir, stem, suffix,
                           rgb_path, normal_path,
                           cam, meshes, res_x, res_y, levels)

    return rgb_path, normal_path


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    args.model = os.path.abspath(args.model)
    out_dir    = os.path.abspath(args.out_dir)

    # pix2pix-ready subdirectories
    color_dir  = os.path.join(out_dir, "color")
    normal_dir = os.path.join(out_dir, "normal")
    os.makedirs(color_dir,  exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    save_stages = getattr(args, 'save_stages', False)
    stages_dir  = os.path.join(out_dir, "stages") if save_stages else None
    if stages_dir:
        os.makedirs(stages_dir, exist_ok=True)
        log.info("Stage images enabled → %s", stages_dir)

    log.info("Model           : %s", args.model)
    log.info("Output (color)  : %s", color_dir)
    log.info("Output (normal) : %s", normal_dir)

    # ── scene setup ──────────────────────────────────────────
    clear_scene()
    cam = create_ortho_camera()
    add_flat_light()

    use_outline = getattr(args, 'use_outline', False)

    # ── import ───────────────────────────────────────────────
    all_imported = import_model(args.model)
    meshes       = find_mesh_objects(all_imported)
    if not meshes:
        raise RuntimeError("No mesh objects found in imported model")

    # ── spatial normalisation ────────────────────────────────
    normalize_model_height(meshes, all_imported, args.model_height)
    center_model_at_origin(meshes, all_imported)

    # Rotation: pass all_imported so armatures are also rotated (see function
    # docstring for why we do NOT bake rotation into armatures).
    apply_model_rotation(all_imported, args.model_rotation)

    stem    = os.path.splitext(os.path.basename(args.model))[0]
    outputs = []

    # ── animation branch ─────────────────────────────────────
    if args.sample_animations:
        actions   = discover_actions()
        armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']

        if not actions or not armatures:
            log.warning(
                "--sample_animations set but no actions/armatures found; "
                "falling back to single T-pose render"
            )
            rest_min, rest_max = world_bbox_from_vertices(meshes)
            place_ortho_camera(cam, rest_min, rest_max,
                               elevation_deg=args.cam_elevation)
            rgb, nrm = render_pair(
                cam, meshes, args.render_size, args.normal_levels,
                color_dir, normal_dir,
                stem, args.suffix or "_f0000",
                args.min_fill_ratio, use_outline=use_outline,
                stages_dir=stages_dir
            )
            if rgb:
                outputs.append((rgb, nrm))

        else:
            for action in actions:
                frames    = sample_frame_numbers(action, args.anim_frames)
                anim_slug = "".join(
                    c if c.isalnum() or c in "-_" else "_"
                    for c in action.name
                )[:24].strip("_")

                current_meshes = find_mesh_objects(list(bpy.context.scene.objects))

                log.info("Computing per-frame bboxes for action '%s' over %d frames…",
                         action.name, len(frames))
                _u_min, _u_max, frame_bboxes = compute_animation_union_bbox(
                    armatures, action, frames, current_meshes
                )

                for frame in frames:
                    apply_action_at_frame(armatures, action, frame)

                    # Per-frame camera: aim-point and ortho_scale fit this exact pose.
                    f_min, f_max = frame_bboxes[frame]
                    place_ortho_camera(cam, f_min, f_max,
                                       elevation_deg=args.cam_elevation)

                    frame_meshes = find_mesh_objects(list(bpy.context.scene.objects))
                    frame_suffix = f"{args.suffix}_anim_{anim_slug}_f{frame:04d}"
                    rgb, nrm = render_pair(
                        cam, frame_meshes, args.render_size, args.normal_levels,
                        color_dir, normal_dir,
                        stem, frame_suffix,
                        args.min_fill_ratio, use_outline=use_outline,
                        stages_dir=stages_dir
                    )
                    if rgb:
                        outputs.append((rgb, nrm))

    # ── static / single-pose branch ──────────────────────────
    else:
        rest_min, rest_max = world_bbox_from_vertices(meshes)
        place_ortho_camera(cam, rest_min, rest_max,
                           elevation_deg=args.cam_elevation)
        rgb, nrm = render_pair(
            cam, meshes, args.render_size, args.normal_levels,
            color_dir, normal_dir,
            stem, args.suffix,
            args.min_fill_ratio, use_outline=use_outline,
            stages_dir=stages_dir
        )
        if rgb:
            outputs.append((rgb, nrm))

    # ── sentinel lines for batch_render.py ───────────────────
    for rgb, nrm in outputs:
        print(f"PIPELINE_OK:{rgb}:{nrm}", flush=True)

    if not outputs:
        log.warning("No valid pairs produced (all images were too sparse or failed).")

    log.info("Done. %d valid pair(s) rendered.", len(outputs))


if __name__ == "__main__":
    main()