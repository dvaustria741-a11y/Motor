#!/usr/bin/env python3
"""Converts the heavy Sketchfab GLBs into compact flat-color LOD meshes (.mdl) for the game.
usage: python3 tools/convert_models.py <glb dir> <out dir>
MDL: 'MDL1', u32 nMeshes; per mesh: u8 kind(0 body,1 glass,2 wheel,3 light,4 normal,5 emissive), u8 r,g,b,a,
     u32 nVerts, u32 nIdx, verts(float px,py,pz, int8 nx,ny,nz,pad), idx(u16)"""
import sys, struct, numpy as np, trimesh, pyfqmr
src, out = sys.argv[1], sys.argv[2]

def mat_color(geo):
    m = getattr(geo.visual, 'material', None)
    try:
        if hasattr(m, 'image') and m.image is not None:
            a = np.asarray(m.image.convert('RGBA').resize((8, 8))).reshape(-1, 4).mean(0)
            return a
        c = getattr(m, 'baseColorFactor', None)
        if c is not None: return np.array(c, float) if np.max(c) > 1 else np.array(c) * 255
        return np.array(geo.visual.main_color, float)
    except Exception:
        return np.array([160, 160, 160, 255], float)

def decimate(v, f, target):
    if len(f) <= target: return v, f
    s = pyfqmr.Simplify(); s.setMesh(v, f); s.simplify_mesh(target_count=max(target, 8), aggressiveness=7, preserve_border=True)
    nv, nf, _ = s.getMesh()
    used, inv = np.unique(nf.reshape(-1), return_inverse=True)
    nv, nf = nv[used], inv.reshape(-1, 3).astype(np.int32)
    if len(nf) > target * 1.4:  # quadric stalled (non-manifold) -> vertex clustering
        ext = (nv.max(0) - nv.min(0)).max(); cell = ext / 400
        for _ in range(14):
            key = np.floor((nv - nv.min(0)) / cell).astype(np.int64)
            _, first, inv = np.unique(key[:, 0] * 1000003 * 1000003 + key[:, 1] * 1000003 + key[:, 2], return_index=True, return_inverse=True)
            cv = np.zeros((inv.max() + 1, 3)); np.add.at(cv, inv, nv); cnt = np.bincount(inv)[:, None]; cv /= cnt
            cf = inv[nf]; ok = (cf[:, 0] != cf[:, 1]) & (cf[:, 1] != cf[:, 2]) & (cf[:, 0] != cf[:, 2]); cf = cf[ok]
            if len(cf) <= target * 1.2 or cell > ext / 4: break
            cell *= 1.3
        u, i2 = np.unique(cf.reshape(-1), return_inverse=True)
        return cv[u], i2.reshape(-1, 3).astype(np.int32)
    return nv, nf

def pack(items, path):
    buf = bytearray(b'MDL1' + struct.pack('<I', len(items)))
    for kind, col, v, f in items:
        m = trimesh.Trimesh(v, f, process=False); m.fix_normals() if False else None
        n = m.vertex_normals if len(f) else np.zeros_like(v)
        nb = np.clip(np.round(n * 127), -127, 127).astype(np.int8)
        buf += struct.pack('<BBBBBII', kind, *[int(max(0, min(255, c))) for c in col[:4]], len(v), len(f) * 3)
        for i in range(len(v)):
            buf += struct.pack('<fff', *v[i]) + struct.pack('<bbbb', *nb[i], 0)
        buf += np.asarray(f, np.uint16).tobytes()
    open(path, 'wb').write(bytes(buf)); print(path, len(buf) // 1024, 'KB')

def world_geoms(scene):
    for n in scene.graph.nodes_geometry:
        T, gn = scene.graph[n]; g = scene.geometry[gn].copy(); g.apply_transform(T); yield n, g

def merge_by(items):  # items: list of (kind,col,v,f) -> merge same (kind,color)
    d = {}
    for k, c, v, f in items:
        key = (k, tuple(int(x) for x in c)); d.setdefault(key, []).append((v, f))
    res = []
    for (k, c), lst in d.items():
        vs, fs, off = [], [], 0
        for v, f in lst: vs.append(v); fs.append(f + off); off += len(v)
        res.append((k, c, np.vstack(vs), np.vstack(fs)))
    return res

# ---------------- BIKE ----------------
import json, re
def glb_json(path):
    d = open(path, 'rb').read(); l = struct.unpack('<I', d[12:16])[0]; return json.loads(d[20:20 + l])
bj = glb_json(f'{src}/suzuki_gsx_750_bike_3d_model.glb')
BM = {m['name']: m for m in bj['materials']}
def bike_mat(name):
    """returns (kind, rgba) -> kind 0 = paint (tinted at runtime), 1 glass, 4 normal, 5 emissive"""
    m = BM.get(name, {}); em = m.get('emissiveFactor'); pbr = m.get('pbrMetallicRoughness', {}).get('baseColorFactor')
    if em and max(em) > 0.5: return 5, [em[0] * 255, em[1] * 255, em[2] * 255, 255]
    if name.startswith('Car_Paint_-_Red'): return 0, [255, 255, 255, 255]
    if name.startswith('Car_Paint_-_Black'): return 4, [22, 22, 26, 255]
    mh = re.match(r'Matte__([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})([0-9A-F]{2})', name)
    if 'trans' in name and mh:
        a, r, g, b = [int(x, 16) for x in mh.groups()]; return 1, [r * .4, g * .4, b * .5, max(a, 90)]
    if pbr: return 4, [c * 255 for c in pbr]
    if 'dourado' in name or 'gold' in name: return 4, [190, 150, 60, 255]
    if 'branco' in name: return 4, [235, 235, 235, 255]
    if mh:
        a, r, g, b = [int(x, 16) for x in mh.groups()]; return 4, [r, g, b, 255]
    return 4, [90, 90, 95, 255]
parts = []
for n, g in world_geoms(sc if False else trimesh.load(f'{src}/suzuki_gsx_750_bike_3d_model.glb')):
    parts.append((g, bike_mat(getattr(g.visual.material, 'name', ''))))
tot = sum(len(g.faces) for g, _ in parts)
for name, budget in (('bike_hi', 14000), ('bike_mid', 6000), ('bike_low', 2200)):
    items = []
    for g, (k, c) in parts:
        t = int(budget * len(g.faces) / tot)
        if len(g.faces) < 40: t = len(g.faces)
        v, f = decimate(np.asarray(g.vertices, np.float64), np.asarray(g.faces, np.int32), max(t, 12))
        if len(f) == 0: continue
        items.append((k, c, v.astype(np.float32), f))
    pack(merge_by(items), f'{out}/{name}.mdl')

# ---------------- CARS ----------------
sc = trimesh.load(f'{src}/generic_passenger_car_pack.glb')
bodies = {}
for n, g in world_geoms(sc):
    base = n.split(' Body')[0].split(' body')[0].split('_')[0]
    if n.startswith('Wheel') or n.startswith('Cylinder'):
        continue
    if 'Body_Body' in n or 'body_Body' in n: key = n.split('_Body')[0].strip()
    else: key = n.split('_Glass')[0].split('_Optics')[0].split('_Body')[0].strip()
    bodies.setdefault(key, []).append((n, g))
wheels = [(n, g) for n, g in world_geoms(sc) if n.startswith('Wheel')]
for key, lst in bodies.items():
    items = []
    allb = trimesh.util.concatenate([g for _, g in lst]); cen = allb.bounds.mean(0); cen[1] = allb.bounds[0][1]
    # wheels belonging to this car: nearest wheel groups by centroid proximity
    near = [g for n, g in wheels if np.linalg.norm((g.centroid - cen)[[0, 2]]) < 2.4 and abs(g.centroid[1]) < 1]
    for n, g in lst:
        kind = 1 if '_Glass' in n else 3 if '_Optics' in n else 0
        col = mat_color(g)
        if kind == 0: col = np.array([255, 255, 255, 255])  # tinted at runtime
        if kind == 1: col = np.array([15, 20, 26, 200])
        if kind == 3: col = np.array([255, 230, 190, 255])
        items.append((kind, col, g.vertices.copy(), g.faces.copy()))
    for g in near: items.append((2, np.array([22, 22, 24, 255]), g.vertices.copy(), g.faces.copy()))
    # recentre: x,z centre, y floor
    allv = np.vstack([i[2] for i in items]); c0 = allv.min(0); c1 = allv.max(0)
    shift = np.array([(c0[0] + c1[0]) / 2, c0[1], (c0[2] + c1[2]) / 2])
    items = [(k, c, (v - shift).astype(np.float32), f) for k, c, v, f in items]
    safe = key.lower().replace(' ', '')
    pack(merge_by(items), f'{out}/car_{safe}.mdl')
    low = []
    tt = sum(len(i[3]) for i in items)
    for k, c, v, f in items:
        v2, f2 = decimate(v.astype(np.float64), f.astype(np.int32), max(int(1500 * len(f) / tt), 10))
        low.append((k, c, v2.astype(np.float32), f2))
    pack(merge_by(low), f'{out}/car_{safe}_low.mdl')

# ---------------- BUILDINGS ----------------
sc = trimesh.load(f'{src}/asian_themed_low_poly_night_city_buildings.glb')
items = []
for n, g in world_geoms(sc):
    if 'FLARE' in n.upper() or 'Flare' in n: continue
    items.append((n, g))
for i, (n, g) in enumerate(items):
    c0, c1 = g.bounds
    ext = c1 - c0
    shift = np.array([(c0[0] + c1[0]) / 2, c0[1], (c0[2] + c1[2]) / 2])
    v = (g.vertices - shift)
    col = mat_color(g)
    print('building', i, n[:30], 'ext', ext.round(1).tolist(), 'tris', len(g.faces), col.round().tolist())
    pack([(4, col, v.astype(np.float32), g.faces.copy())], f'{out}/bld_{i}.mdl')
