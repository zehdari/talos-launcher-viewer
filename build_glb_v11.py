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
    if n.startswith('SHCS 4-40 x 1-2 (92196A110)'): return 7, u(-1 if n.endswith('-1') else 1, 0, 0), 35   # marker spring retention, side
    if n.startswith('SHCS 4-40 x 5-8') and not n.endswith('-1'): return 4, u(-1 if n.endswith('-2') else 1, 0, 0), 40   # spring retention, from the side
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
                     ('Marker Shaft', (6, u(0, -1, 0), 100)), ('Left Helix', (6, u(0, 1, 0), 60)),
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
def category(n):
    if any(k in n for k in ('Torpedo Body', 'Marker Base', 'Top Cover', 'Bottom Cover', 'Crossbar Arm', 'Servo Holder')): return 'Structure'
    if any(k in n for k in ('Torpedo Shaft', 'Pushing Arm', 'Marker Shaft', 'Helix', 'Servogear', 'SER-20XX')): return 'Drivetrain'
    if any(k in n for k in ('Torpedo D18', 'Marker P83', 'Spring')): return 'Payloads + springs'
    return 'Fasteners'
def spring(n):
    m = re.match(r'LV11 Spring 9435K123 \((torpedo|marker), loaded [0-9.]+\)-(\d)', n)
    if not m: return None
    return dict(kind=m.group(1), side=('L' if m.group(2) == '1' else 'R') if m.group(1) == 'torpedo' else ('left' if m.group(2) == '1' else 'right'))
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
                      label=re.sub(r'-\d+$', '', comp), inst=int(re.search(r'-(\d+)$', comp).group(1)), cat=category(comp), spring=spring(comp))
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
# ---------- assembly procedure: one exact part per step (several instances of it allowed) ----------
# (old coarse step, label, instance filter or None, caption)
MB, PA, HD, XA = 'LV11 Marker Base', 'LV11 Pushing Arm', 'LV11 Servo Holder', 'LV11 Crossbar Arm'
# (old coarse step, label, instance filter or None, caption, sub-assembly host or None) ; ('MOUNT', host, caption)
SEQ = [(0, 'LV11 Torpedo Body', None, 'Torpedo body: the frame everything mounts to.', None),
       (1, 'Hex Nut 4-40 (91841A005)', None, 'Captive 4-40 nuts into the body traps (top cover x4, marker base x3, bottom cover screw D).', None),
       (1, 'Hex Nut 8-32', {1, 2}, 'Captive 8-32 nuts into the body traps for the servo-holder ears.', None),
       (2, 'Right Helix Modified', None, 'Right helix into its cavity through the marker-base opening (bigger than the bearings, so it goes in before its shaft).', None),
       (2, 'LV11 Torpedo Shaft', None, 'Printed torpedo shaft (18T gear end stays below) up through both bearings and the right helix; pin the helix.', None),
       (3, PA, None, 'Sub-assembly on the bench: pushing arm.', PA),
       (3, 'Hex Nut 4-40 (91841A005)', {10}, 'Cross-pin nut into the pushing arm pocket.', PA),
       ('MOUNT', PA, 'Pushing arm onto the shaft top, down to the thrust face.'),
       (3, 'SHCS 4-40 x 5-8 (92196A112)', {1}, 'Cross-pin screw through the arm and shaft into the nut.', None),
       (4, 'LV11 Spring 9435K123 (torpedo, loaded 22.41)', None, 'Torpedo springs onto their guide pins.', None),
       (4, 'SHCS 4-40 x 5-8 (92196A112)', {2, 3}, "Spring retention cap screws 4-40 x 5/8 in from the outboard side onto the spotface; the tip presses the spring's closed end coil.", None),
       (5, 'LV11 Top Cover', None, 'Top cover onto the body.', None),
       (5, 'FHMS 4-40 x 1 100deg', None, 'Flat-head 4-40 x 1 screws through the top cover into the captive nuts.', None),
       (6, MB, None, 'Sub-assembly on the bench: marker base.', MB),
       (6, 'Hex Nut 4-40 (91841A005)', {8}, 'Captive 4-40 nut into the marker base for bottom cover screw C.', MB),
       (6, 'LV11 Marker Shaft', None, 'Marker shaft (lobe end down) up through the marker-base bearing.', MB),
       (6, 'Left Helix Modified', None, 'Left helix onto the top of the marker shaft; pin it.', MB),
       ('MOUNT', MB, 'Marker base sub-assembly up to the body: the left helix enters the cavity and meshes with the right helix.'),
       (6, 'SHCS 4-40 x 1-1-2', None, 'Socket-head 4-40 x 1-1/2 screws up through the marker base into the body nuts.', None),
       (7, 'LV11 Spring 9435K123 (marker, loaded 27.46)', None, 'Marker springs onto their guide pins.', None),
       (7, 'SHCS 4-40 x 1-2 (92196A110)', None, "Marker spring retention cap screws 4-40 x 1/2 in from the sides of the marker base; the tip presses the spring's end coil.", None),
       (7, 'LV11 Bottom Cover', None, 'Bottom cover onto the marker base.', None),
       (7, 'FHSCS 4-40 x 2 100deg', None, 'Bottom cover screw C (4-40 x 2 flat head) into the marker-base nut.', None),
       (7, 'FHMS 4-40 x 2-1-2 100deg', None, 'Bottom cover screw D (4-40 x 2-1/2 flat head) through the marker base into the body nut.', None),
       (8, HD, None, 'Sub-assembly on the bench: servo holder.', HD),
       (9, 'SER-20XX Servo', None, 'SER-20XX servo onto its seat with 4 M3 through-bolts.', HD),
       (9, 'Servogear', None, 'Servo gear onto the spline.', HD),
       ('MOUNT', HD, "Servo holder sub-assembly onto the body's rear face: the servo gear slides into mesh with the shaft gear."),
       (8, 'SHCS 8-32 x 3-4', None, 'Socket-head 8-32 x 3/4 screws through the ears into the body nuts.', None),
       (10, XA, None, 'Sub-assembly on the bench: crossbar arm.', XA),
       (10, 'Hex Nut 8-32', {5, 6}, 'Crossbar captive nuts into the side slots (crossbar screws go in on the robot).', XA),
       (10, 'Hex Nut 8-32', {3, 4}, 'Fork nuts into their recesses.', XA),
       ('MOUNT', XA, 'Crossbar arm sub-assembly onto the body keel.'),
       (10, 'SHCS 8-32 x 2-1-2', None, 'Fork bolts 8-32 x 2-1/2 through the fork and keel into the nuts.', None),
       (11, 'LV11 Torpedo D18', None, 'Load the torpedoes through the top cover; twist to lock.', None),
       (11, 'LV11 Marker P83', None, 'Load the markers up into the bottom cover; twist to lock.', None)]
def newstep(m):
    for i, e in enumerate(SEQ):
        if e[0] == 'MOUNT': continue
        old_st, lab, inst, cap, sub = e
        if m['label'] == lab and m['step'] == old_st and (inst is None or m['inst'] in inst): return i, sub
    raise SystemExit('no step for ' + m['label'] + ' #' + str(m['inst']))
for k_, v in meta.items():
    if 'step' in v: v['step_old'] = v['step']; v['step'], v['sub'] = newstep(v)
STEPS = [e[2] if e[0] == 'MOUNT' else e[3] for e in SEQ]
STEP_OF = {e[1]: i for i, e in enumerate(SEQ) if e[0] != 'MOUNT'}
BENCH = {MB: 170, PA: 120, HD: 160, XA: 170}      # bench distance along the host's approach direction
host_ap = {v['label']: np.array(v['approach']) for v in meta.values() if v.get('label') in BENCH}
SUBS = {h: dict(mount=next(i for i, e in enumerate(SEQ) if e[0] == 'MOUNT' and e[1] == h),
                bench=[round(float(x), 3) for x in host_ap[h] / np.linalg.norm(host_ap[h]) * BENCH[h]]) for h in BENCH}
old = json.load(open(OUT / 'parts.json'))
meta.update({k: v for k, v in old.items() if k.startswith(('robot__', 'v8__', 'solo__', 'rail'))})
json.dump(meta, open(OUT / 'parts.json', 'w'), indent=0)
TO = 26 * math.sqrt(0.5)
# spring launch physics (McMaster 9435K123: free 38.1 mm; torpedo spring loaded 22.41 holds 0.41 J -> k)
FREE, LT, LM = 38.1, 22.41, 27.46
k = 2 * 0.41 / ((FREE - LT) / 1000) ** 2
E_m = 0.5 * k * ((FREE - LM) / 1000) ** 2
def coast(m_kg, d_mm, cd): return 0.5 * 1000 * cd * math.pi * (d_mm / 2000) ** 2 / m_kg      # drag c [1/m]: dv/dt = -c v^2
mt, mm = 0.0358 * 1.1, 0.0127 * 1.1                                                             # + ~10 % added mass in water
phys = dict(slow=25, k=k, torpedo=dict(stroke=FREE - LT, loaded=LT, v0=math.sqrt(2 * 0.41 / mt), c=coast(mt, 18.0, 0.4), tq=math.pi / 2 * math.sqrt(mt / k), base_z=O[2] + 2.918),
            marker=dict(stroke=FREE - LM, loaded=LM, v0=math.sqrt(2 * E_m / mm), c=coast(mm, 13.2, 0.5), tq=math.pi / 2 * math.sqrt(mm / k), top_y=O[1] - 32.98, sink=0.15))
kin = dict(O=O.tolist(), steps=STEPS, subs=SUBS, phys=phys,
           axes=dict(shaft=[O[0], O[1]], servo=[O[0], O[1] - 22.5], marker=[O[0], O[2] - 23.0]),
           ratio_servo_to_shaft=-27 / 18,          # shaft angle = -1.5 x servo angle (external spur)
           torpedo=dict(L=dict(axis=[O[0] - TO, O[1] + TO], release=51.5, unlock=47),
                        R=dict(axis=[O[0] + TO, O[1] + TO], release=111.5, unlock=47)),
           marker=dict(right=dict(axis=[O[0] + 17.5, O[2] - 40.5], release=-49.75, lock=-40.685),
                       left=dict(axis=[O[0] - 17.5, O[2] - 40.5], release=-109.75, lock=-41.54)))
json.dump(kin, open(OUT / 'kin.json', 'w'), indent=1)
print('v11.glb', round((OUT / 'v11.glb').stat().st_size / 1e6, 2), 'MB; torps.glb', round((OUT / 'torps.glb').stat().st_size / 1e6, 2), 'MB;',
      sum(1 for k in meta if k.startswith('v11__')), 'launcher nodes')

# ---------- BOM (counts from the assembly; links: printed -> print-ready STEP/STL in this repo, McMaster) ----------
MC = 'https://www.mcmaster.com/'
CAT = dict(shcs=MC + 'products/socket-head-screws/', fh=MC + 'products/flat-head-screws/', nut=MC + 'products/hex-nuts/')
BOMDEF = {   # assembly label -> (display name, kind, spec, link or None (printed -> print files), note)
    'LV11 Torpedo Body': ('Torpedo Body', 'printed', 'PETG, 100 % infill', None, 'holder face on the bed'),
    'LV11 Marker Base': ('Marker Base', 'printed', 'PETG, 100 % infill', None, 'joint face on the bed'),
    'LV11 Top Cover': ('Top Cover (open frame)', 'printed', 'PETG, 100 % infill', None, 'exit face on the bed; arrows = AMS colour'),
    'LV11 Bottom Cover': ('Bottom Cover', 'printed', 'PETG, 100 % infill', None, 'exit face on the bed; arrows = AMS colour'),
    'LV11 Crossbar Arm': ('Crossbar Arm', 'printed', 'PETG, 100 % infill', None, 'crossbar seat on the bed'),
    'LV11 Servo Holder': ('Servo Holder', 'printed', 'PETG, 100 % infill', None, 'on its side'),
    'LV11 Pushing Arm': ('Pushing Arm', 'printed', 'PETG, 100 % infill', None, 'thrust face on the bed'),
    'LV11 Torpedo Shaft': ('Torpedo Shaft with 18T gear', 'printed', 'PETG, 100 % infill', None, 'gear face on the bed'),
    'LV11 Marker Shaft': ('Marker Shaft', 'printed', 'PETG, 100 % infill', None, 'lobe end on the bed'),
    'Servogear': ('Servo Gear 27T (team)', 'printed', 'PETG', None, 'team part, unchanged'),
    'Right Helix Modified': ('Right Helix (team)', 'printed', 'PETG', None, 'team part, unchanged'),
    'Left Helix Modified': ('Left Helix (team)', 'printed', 'PETG', None, 'team part, unchanged'),
    'LV11 Torpedo D18': ('Torpedo D18', 'printed', 'PETG, 100 % infill', None, 'tail on the bed'),
    'LV11 Marker P83': ('Marker P83', 'printed', 'PETG, 100 % infill', None, 'tail on the bed'),
    'SER-20XX Servo': ('Underwater servo SER-20XX (Blue Trail)', 'purchased', 'SER-20XX', 'https://www.bluetrailengineering.com/', 'team stock'),
    'LV11 Spring 9435K123 (torpedo, loaded 22.41)': ('Compression spring 9435K123 (torpedo)', 'McMaster', '9435K123, OD 12.2, free 38.1', MC + '9435K123/', 'loaded 22.41 mm'),
    'LV11 Spring 9435K123 (marker, loaded 27.46)': ('Compression spring 9435K123 (marker)', 'McMaster', '9435K123, OD 12.2, free 38.1', MC + '9435K123/', 'loaded 27.46 mm'),
    'FHMS 4-40 x 1 100deg': ('Flat head screw 4-40 x 1", 100 deg, 18-8', 'McMaster', '4-40 x 1" 100 deg FH', CAT['fh'], 'top cover -> body nuts'),
    'FHMS 4-40 x 2-1-2 100deg': ('Flat head screw 4-40 x 2-1/2", 100 deg, 18-8', 'McMaster', '4-40 x 2-1/2" 100 deg FH', CAT['fh'], 'bottom cover D -> body nut'),
    'FHSCS 4-40 x 2 100deg': ('Flat head screw 4-40 x 2", 100 deg, 18-8', 'McMaster', '4-40 x 2" 100 deg FH', CAT['fh'], 'bottom cover C -> marker base nut'),
    'SHCS 4-40 x 1-1-2': ('Socket head screw 4-40 x 1-1/2", 18-8', 'McMaster', '4-40 x 1-1/2" SHCS', CAT['shcs'], 'marker base -> body'),
    'SHCS 4-40 x 5-8 (92196A112)': ('Socket head screw 4-40 x 5/8", 18-8', 'McMaster', '92196A112', MC + '92196A112/', 'pushing arm cross pin (1) + torpedo spring retention from the side (2)'),
    'SHCS 4-40 x 1-2 (92196A110)': ('Socket head screw 4-40 x 1/2", 18-8', 'McMaster', '92196A110', MC + '92196A110/', 'marker spring retention, sides of the marker base'),
    'Hex Nut 4-40 (91841A005)': ('Hex nut 4-40, 18-8', 'McMaster', '91841A005', MC + '91841A005/', 'captive in traps'),
    'SHCS 8-32 x 3-4': ('Socket head screw 8-32 x 3/4", 18-8', 'McMaster', '8-32 x 3/4" SHCS', CAT['shcs'], 'servo holder ears -> body'),
    'SHCS 8-32 x 2-1-2': ('Socket head screw 8-32 x 2-1/2", 18-8', 'McMaster', '8-32 x 2-1/2" SHCS', CAT['shcs'], 'crossbar arm fork -> keel'),
    'Hex Nut 8-32': ('Hex nut 8-32, 18-8', 'McMaster', '8-32 hex nut', CAT['nut'], 'captive in traps'),
}
EXTRA = [  # hardware the launcher assembly does not model
    ('Socket head screw 8-32 x 3/8", 18-8', 'McMaster', '92196A192', MC + '92196A192/', 2, 'front crossbar -> crossbar arm nuts (robot fit)', 10),
    ('Screw #8 + nut (team stock)', 'McMaster', '#8, length to suit', CAT['shcs'], 2, 'rear crossbar -> servo holder slab', 10),
    ('Socket head screw M3 x 45 + nut', 'McMaster', 'M3 x 45', CAT['shcs'], 4, 'through the servo body (Blue Trail side mount)', 9),
    ('Right helix cross pin screw #4-40 + nut', 'McMaster', '4-40, length to suit', CAT['shcs'], 1, 'team pin through the right helix and torpedo shaft', 'RH'),
    ('Left helix cross pin screw #4-40 + nut', 'McMaster', '4-40, length to suit', CAT['shcs'], 1, 'team pin through the left helix and marker shaft', 'LH'),
]
from collections import Counter, defaultdict
cnt, steps = Counter(), defaultdict(set)
for k, v in meta.items():
    if v.get('label') in BOMDEF: cnt[v['label']] += 1; steps[v['label']].add(v['step'])
files = {}
for f in sorted((W / 'v11/export/print').glob('* PRINT.step')):
    stem = f.name.replace(' PRINT.step', ''); files[stem] = stem
bom = []
for lab, (name, kind, spec, link, note) in BOMDEF.items():
    stem = lab.replace('reused/', '')
    row = dict(name=name, kind=kind, spec=spec, qty=cnt[lab], note=note, steps=sorted(steps[lab]), label=lab)
    if kind == 'printed' and stem in files:
        row['step_file'] = f'print/{stem} PRINT.step'; row['stl_file'] = f'print/{stem} PRINT.STL'
    else: row['link'] = link
    bom.append(row)
EXTRA_AT = {10: [STEP_OF['Hex Nut 8-32']] if False else [next(i for i, e in enumerate(SEQ) if e[0] != 'MOUNT' and e[1] == 'Hex Nut 8-32' and e[2] == {5, 6})], 9: [STEP_OF['SER-20XX Servo']], 'RH': [STEP_OF['LV11 Torpedo Shaft']], 'LH': [STEP_OF['Left Helix Modified']]}
for name, kind, spec, link, q, note, st in EXTRA:
    bom.append(dict(name=name, kind=kind, spec=spec, qty=q, note=note + ' (not modelled)', steps=EXTRA_AT[st], link=link, label=None))
json.dump(bom, open(OUT / 'bom.json', 'w'), indent=1)
print('bom rows', len(bom), 'printed', sum(r['qty'] for r in bom if r['kind'] == 'printed'))
