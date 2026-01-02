#pip install pillow numpy
import math
import numpy as np
from PIL import Image, ImageDraw

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(n == 0, 1.0, n)


def platonic_solid(name: str):
    name = name.lower().strip()

    if name == "tetra":
        V = np.array([
            ( 1,  1,  1),
            ( 1, -1, -1),
            (-1,  1, -1),
            (-1, -1,  1),
        ], dtype=float)
        F = [[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]]
        return {"V": normalize(V), "F": F}

    if name == "cube":
        V = np.array([
            (-1, -1, -1), ( 1, -1, -1), ( 1,  1, -1), (-1,  1, -1),
            (-1, -1,  1), ( 1, -1,  1), ( 1,  1,  1), (-1,  1,  1),
        ], dtype=float)
        F = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 4, 7, 3],
            [1, 5, 6, 2],
            [3, 2, 6, 7],
            [0, 1, 5, 4],
        ]
        return {"V": normalize(V), "F": F}

    if name == "octa":
        V = np.array([
            ( 1, 0, 0), (-1, 0, 0),
            ( 0, 1, 0), ( 0,-1, 0),
            ( 0, 0, 1), ( 0, 0,-1),
        ], dtype=float)
        F = [
            [4, 0, 2], [4, 2, 1], [4, 1, 3], [4, 3, 0],
            [5, 2, 0], [5, 1, 2], [5, 3, 1], [5, 0, 3],
        ]
        return {"V": normalize(V), "F": F}

    if name == "icosa":
        phi = (1 + math.sqrt(5)) / 2
        V = np.array([
            (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
            ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
            ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
        ], dtype=float)
        F = [
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
        ]
        return {"V": normalize(V), "F": F}

    if name == "dodeca":
        ico = platonic_solid("icosa")
        V_ico = ico["V"]
        F_ico = ico["F"]

        dual_V = []
        for face in F_ico:
            dual_V.append(V_ico[np.array(face)].mean(axis=0))
        V = normalize(np.array(dual_V, dtype=float))  # 20 vertices

        F = []
        for v_idx in range(V_ico.shape[0]):
            adjacent = [fi for fi, face in enumerate(F_ico) if v_idx in face]
            v = V_ico[v_idx]

            tmp = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(tmp, v)) > 0.9:
                tmp = np.array([0.0, 1.0, 0.0])
            u = np.cross(v, tmp); u /= (np.linalg.norm(u) + 1e-9)
            w = np.cross(v, u)

            ang = []
            for fi in adjacent:
                p = V[fi]
                ang.append((math.atan2(np.dot(p, w), np.dot(p, u)), fi))
            ang.sort()
            F.append([fi for _, fi in ang])

        return {"V": V, "F": F}

    raise ValueError("Solid must be one of: tetra, cube, octa, dodeca, icosa")

#rendering

def rot_y(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

def rot_x(t: float) -> np.ndarray:
    c, s = math.cos(t), math.sin(t)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

def project(P: np.ndarray, size: int, fov: float, camdist: float):
    z = P[:, 2] + camdist
    f = (size / 2) / math.tan(fov / 2)
    x = (P[:, 0] * f) / (z + 1e-9) + size / 2
    y = (-P[:, 1] * f) / (z + 1e-9) + size / 2
    return np.stack([x, y], axis=1), z

def triangulate(face):
    if len(face) == 3:
        return [face]
    return [[face[0], face[i], face[i + 1]] for i in range(1, len(face) - 1)]

def affine_coeffs(dst_tri, src_tri):
    A, B = [], []
    for (xd, yd), (xs, ys) in zip(dst_tri, src_tri):
        A.append([xd, yd, 1, 0, 0, 0])
        A.append([0, 0, 0, xd, yd, 1])
        B.append(xs)
        B.append(ys)
    sol = np.linalg.solve(np.array(A, float), np.array(B, float))
    return tuple(sol.tolist())

#uvs

def _proj_to_plane(vec: np.ndarray, n: np.ndarray) -> np.ndarray:
    return vec - n * float(np.dot(vec, n))

def standardized_face_uvs(face, V_rest):

    idx = np.array(face, dtype=int)
    pts = V_rest[idx]
    c = pts.mean(axis=0)

    n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
    n /= (np.linalg.norm(n) + 1e-9)

    ref_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ]
    u_axis = None
    for ref in ref_axes:
        cand = _proj_to_plane(ref, n)
        if np.linalg.norm(cand) > 1e-6:
            u_axis = cand / np.linalg.norm(cand)
            break
    if u_axis is None:
        u_axis = np.array([1.0, 0.0, 0.0])

    v_axis = np.cross(n, u_axis)
    v_axis /= (np.linalg.norm(v_axis) + 1e-9)

    # enforce consistent up direction
    up_ref = _proj_to_plane(np.array([0.0, 1.0, 0.0]), n)
    if np.linalg.norm(up_ref) > 1e-6:
        up_ref /= np.linalg.norm(up_ref)
        if float(np.dot(v_axis, up_ref)) < 0:
            u_axis = -u_axis
            v_axis = -v_axis

    rel = pts - c
    u = rel @ u_axis
    v = rel @ v_axis

    umin, umax = float(u.min()), float(u.max())
    vmin, vmax = float(v.min()), float(v.max())
    du = (umax - umin) if (umax - umin) > 1e-9 else 1.0
    dv = (vmax - vmin) if (vmax - vmin) > 1e-9 else 1.0

    U = 1.0 - (u - umin) / du
    V = 1.0 - (v - vmin) / dv
    return list(zip(U.tolist(), V.tolist()))

def precompute_uvs(solid):
    V_rest = solid["V"]
    return [standardized_face_uvs(face, V_rest) for face in solid["F"]]

#frames

def render_frame(tex: Image.Image, solid, face_uvs_list, size: int, theta: float, pitch: float, fov: float, camdist: float):
    tex = tex.convert("RGBA")
    W, H = tex.size

    R = rot_y(theta) @ rot_x(pitch)
    V3 = solid["V"] @ R.T
    pts2d, z = project(V3, size, fov, camdist)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    tri_items = []
    for fi, face in enumerate(solid["F"]):
        uvs = face_uvs_list[fi]
        for t in triangulate(face):
            local = [face.index(t[0]), face.index(t[1]), face.index(t[2])]
            tri_uv = [uvs[local[0]], uvs[local[1]], uvs[local[2]]]
            depth = float((z[t[0]] + z[t[1]] + z[t[2]]) / 3.0)
            tri_items.append((t, tri_uv, depth))

    tri_items.sort(key=lambda x: x[2], reverse=True)

    for (i, j, k), tri_uv, _ in tri_items:
        dst = [tuple(pts2d[i]), tuple(pts2d[j]), tuple(pts2d[k])]

        area2 = abs(
            (dst[1][0] - dst[0][0]) * (dst[2][1] - dst[0][1]) -
            (dst[1][1] - dst[0][1]) * (dst[2][0] - dst[0][0])
        )
        if area2 < 1e-2:
            continue

        src = [
            (tri_uv[0][0] * W, tri_uv[0][1] * H),
            (tri_uv[1][0] * W, tri_uv[1][1] * H),
            (tri_uv[2][0] * W, tri_uv[2][1] * H),
        ]

        coeff = affine_coeffs(dst, src)
        warped = tex.transform((size, size), Image.Transform.AFFINE, coeff, resample=Image.Resampling.BICUBIC)

        mask = Image.new("L", (size, size), 0)
        ImageDraw.Draw(mask).polygon(dst, fill=255)
        canvas = Image.composite(warped, canvas, mask)

    return canvas

#user input

def ask(prompt, cast=str, default=None):
    if default is None:
        s = input(f"{prompt}: ").strip()
        if not s:
            raise ValueError(f"{prompt} is required.")
        return cast(s)
    s = input(f"{prompt} [{default}]: ").strip()
    return cast(s) if s else default


def main():
    print("\n--- Platonic Solid Texture Spinner (Inverted Right + Up) ---\n")

    png_path = ask("Input PNG path", str)
    solid_name = ask("Solid (tetra / cube / octa / dodeca / icosa)", str, "cube")
    out_gif = ask("Output GIF filename", str, "spin.gif")
    size = ask("Frame size (px)", int, 512)
    frames = ask("Number of frames", int, 60)
    fps = ask("FPS", int, 20)
    pitch_deg = ask("Tilt angle (degrees)", float, 20.0)
    fov_deg = ask("Field of view (degrees)", float, 55.0)
    camdist = ask("Camera distance", float, 3.0)

    tex = Image.open(png_path).convert("RGBA")
    solid = platonic_solid(solid_name)

    face_uvs_list = precompute_uvs(solid)

    pitch = math.radians(pitch_deg)
    fov = math.radians(fov_deg)

    images = []
    for i in range(frames):
        theta = 2 * math.pi * (i / frames)
        images.append(render_frame(tex, solid, face_uvs_list, size, theta, pitch, fov, camdist))

    duration_ms = int(round(1000 / fps))
    images[0].save(
        out_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=False,
    )

    print(f"\nSaved GIF → {out_gif}\n")


if __name__ == "__main__":
    main()
