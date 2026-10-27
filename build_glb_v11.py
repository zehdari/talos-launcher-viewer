"""Viewer assets for V11: v11.glb (launcher + markers) and torps.glb (D18 torpedoes) from the LV11 Launcher per-component
STL export (launcher frame, rest pose), moved to the robot frame (datum O). parts.json carries, per node: colour, closed,
assembly step + approach vector (robot-frame mm) and the rotation group the firing animation turns (shaft / servo /
marker). steps.json holds the assembly procedure captions and the gear/release kinematics."""
import json, math, re, sys, numpy as np, trimesh, fast_simplification as fs
from pathlib import Path
W = Path('/home/cameron/torpedo_work'); OUT = W / 'viewer_repo'; SRC = W / 'v11/viewer_stl'
sys.path.insert(0, str(W / 'launcher/lib')); from meshutil import clean
O = np.array([600.414, 892.833, 1199.5514])            # launcher datum in the robot frame (LV11 Robot Fit)
COL = [('Torpedo Body', [0.92, 0.55, 0.22]), ('Marker Base', [0.27, 0.51, 0.86]), ('Top Cover', [0.31, 0.69, 0.35]),
       ('Crossbar Arm', [0.65, 0.37, 0.80]), ('Pushing Arm', [0.84, 0.22, 0.22]), ('Servo Holder', [0.43, 0.49, 0.59]),
       ('Torpedo D18', [0.96, 0.80, 0.16]), ('Marker P83', [0.20, 0.78, 0.84]), ('Bottom Cover', [0.55, 0.78, 0.47]),
       ('SER-20XX', [0.12, 0.12, 0.13]), ('Helix', [0.85, 0.68, 0.30]), ('Servogear', [0.85, 0.68, 0.30]),
       ('Torpedo Shaft', [0.95, 0.95, 0.97]), ('Marker Shaft', [0.95, 0.95, 0.97]), ('Spring', [0.80, 0.80, 0.82])]
def colfor(n): return next((c for k, c in COL if k.lower() in n.lower()), [0.62, 0.64, 0.68])
def u(*v): v = np.array(v, float); return v / np.linalg.norm(v)
c15, s15 = math.cos(math.radians(15)), math.sin(math.radians(15))
# component -> (step, approach unit vector, distance mm)
NUT_UA = {1: 90, 2: 113, 3: 23, 4: 293}
def plan(n):
    m = re.match(r'Hex Nut 4-40 \(91841A005\)-(\d+)', n)
    if m:
        i = int(m.group(1))
        if i in NUT_UA: a = math.radians(NUT_UA[i]); return 1, u(math.cos(a), math.sin(a), 0), 25
        if i in (5, 6, 7, 9): return 1, u(0, -1, 0), 30
        if i == 8: return 6, u(0, 1, 0), 30
        if i == 10: return 3, u(-c15, -s15, 0), 25
    m = re.match(r'Hex Nut 8-32-(\d+)', n)
    if m:
        i = int(m.group(1))
        if i in (1, 2): return 1, u(1 if i == 1 else -1, 0, 0), 30
        if i in (3, 4): return 10, u(-1, 0, 0), 35
        if i in (5, 6): return 10, u(1 if i == 6 else -1, 0, 0), 30
    for key, val in (('Right Helix', (2, u(0, -1, 0), 70)), ('Torpedo Shaft', (2, u(0, 0, -1), 130)),
                     ('Pushing Arm', (3, u(0, 0, 1), 70)), ('SHCS 4-40 x 5-8', (3, u(c15, s15, 0), 40)),
                     ('Spring 9435K123 (torpedo', (4, u(0, 0, 1), 90)), ('Top Cover', (5, u(0, 0, 1), 90)),
                     ('FHMS 4-40 x 1 ', (5, u(0, 0, 1), 70)), ('Marker Base', (6, u(0, -1, 0), 100)),
                     ('Marker Shaft', (6, u(0, -1, 0), 100)), ('Left Helix', (6, u(0, -1, 0), 100)),
                     ('SHCS 4-40 x 1-1-2', (6, u(0, -1, 0), 70)), ('Spring 9435K123 (marker', (7, u(0, -1, 0), 70)),
                     ('Bottom Cover', (7, u(0, -1, 0), 90)), ('FHSCS 4-40 x 2', (7, u(0, -1, 0), 70)),
                     ('FHMS 4-40 x 2-1-2', (7, u(0, -1, 0), 70)), ('Servo Holder', (8, u(0, 0, -1), 110)),
                     ('SHCS 8-32 x 3-4', (8, u(0, 0, -1), 50)), ('SER-20XX', (9, u(0, 1, 0), 80)),
                     ('Servogear', (9, u(0, 0, 1), 45)), ('Crossbar Arm', (10, u(0, -1, 0), 100)),
                     ('SHCS 8-32 x 2-1-2', (10, u(1, 0, 0), 70)), ('Torpedo D18', (11, u(0, 0, 1), 170)),
                     ('Marker P83', (11, u(0, -1, 0), 130))):
        if key in n: return val
    if 'Torpedo Body' in n: return 0, u(0, 0, 1), 0
    raise SystemExit('no plan for ' + n)
def rotgroup(n):
    if any(k in n for k in ('Torpedo Shaft', 'Pushing Arm', 'Right Helix', 'SHCS 4-40 x 5-8', 'Hex Nut 4-40 (91841A005)-10')): return 'shaft'
    if 'Servogear' in n: return 'servo'
    if 'Marker Shaft' in n or 'Left Helix' in n: return 'marker'
    return None
STEPS = ["Torpedo body: the frame everything mounts to.",
         "Captive nuts into the body traps (top cover, marker base, servo-holder ears).",
         "Right helix into its cavity, then the printed torpedo shaft (18T gear on it) up through both bearings.",
         "Pushing arm onto the shaft top, 4-40 cross pin + nut.",
         "Torpedo springs onto their guide pins.",
         "Top cover onto the body, 4 flat-head 4-40s into the captive nuts.",
         "Marker shaft + left helix into the marker base, base up to the body, 3 SHCS 4-40.",
         "Marker springs, bottom cover, screws C and D.",
         "Servo holder to the body's rear face, 2 SHCS 8-32 into captive nuts.",
         "SER-20XX servo onto its seat (4 M3 through-bolts), servo gear onto the spline.",
         "Crossbar arm onto the keel: 2 bolts + nuts, crossbar captive nuts in their side slots.",
         "Load: torpedoes through the top cover, markers up into the bottom cover."]
def c8(c): return (np.array(list(c) + [1.0]) * 255).astype(np.uint8)
meta = {}
def add(scene, name, m, color, group, comp, target=40000):
    m = m.copy()
    if target and len(m.faces) > target:
        V, F = fs.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int64), target_reduction=1 - target / len(m.faces))
        m2 = clean(trimesh.Trimesh(V, F, process=False))
        if m2.is_watertight or not m.is_watertight: m = m2
    mat = trimesh.visual.material.PBRMaterial(baseColorFactor=c8(color), metallicFactor=0.1, roughnessFactor=0.6, doubleSided=True)
    m.visual = trimesh.visual.TextureVisuals(material=mat)
    scene.add_geometry(m, node_name=name, geom_name=name)
    st, ap, d = plan(comp)
    meta[name] = dict(group=group, closed=bool(m.is_watertight), color=list(color), faces=len(m.faces),
                      step=st, approach=[round(float(v), 3) for v in ap * d], rot=rotgroup(comp),
                      label=re.sub(r'-\d+$', '', comp))
v11, tp = trimesh.Scene(), trimesh.Scene()
for f in sorted(SRC.glob('*.STL')):
    n = re.sub(r'^v11 - ', '', f.stem)
    m = clean(trimesh.load(f, force='mesh', process=False)); m.apply_translation(O)
    if 'Torpedo D18' in n:
        add(tp, 'tp__Torpedo_D18_' + ('L' if n.endswith('-1') else 'R'), m, colfor(n), 'torps', n, target=60000)
    elif 'Marker P83' in n:
        add(v11, 'v11__Marker_P83_' + ('left' if n.endswith('-1') else 'right'), m, colfor(n), 'v11', n)
    else:
        add(v11, 'v11__' + re.sub(r'[^A-Za-z0-9_]+', '_', n), m, colfor(n), 'v11', n)
v11.export(OUT / 'v11.glb'); tp.export(OUT / 'torps.glb')
old = json.load(open(OUT / 'parts.json'))
meta.update({k: v for k, v in old.items() if k.startswith(('robot__', 'v8__', 'solo__', 'rail'))})
json.dump(meta, open(OUT / 'parts.json', 'w'), indent=0)
TO = 26 * math.sqrt(0.5)
kin = dict(O=O.tolist(), steps=STEPS,
           axes=dict(shaft=[O[0], O[1]], servo=[O[0], O[1] - 22.5], marker=[O[0], O[2] - 23.0]),
           ratio_servo_to_shaft=-27 / 18,          # shaft angle = -1.5 x servo angle (external spur)
           torpedo=dict(L=dict(axis=[O[0] - TO, O[1] + TO], release=51.5, unlock=47),
                        R=dict(axis=[O[0] + TO, O[1] + TO], release=111.5, unlock=47)),
           marker=dict(right=dict(axis=[O[0] + 17.5, O[2] - 40.5], release=-49.75, lock=-40.685),
                       left=dict(axis=[O[0] - 17.5, O[2] - 40.5], release=-109.75, lock=-41.54)))
json.dump(kin, open(OUT / 'kin.json', 'w'), indent=1)
print('v11.glb', round((OUT / 'v11.glb').stat().st_size / 1e6, 2), 'MB; torps.glb', round((OUT / 'torps.glb').stat().st_size / 1e6, 2), 'MB;',
      sum(1 for k in meta if k.startswith('v11__')), 'launcher nodes')
