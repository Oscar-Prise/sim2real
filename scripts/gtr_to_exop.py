"""Re-parent the LGTR / RGTR markers in the exo model from femur_l/femur_r to ExoP, keeping the
global position at the zero-coordinate (default) pose.

At zero coordinates:  femur frame = pelvis frame shifted by the hip joint location,
                      ExoP frame  = pelvis frame shifted by -ExoP_offset translation (Pelvis2ExoP FreeJoint).
So  loc_ExoP = loc_femur + hip_location_in_pelvis + ExoP_offset_translation.
"""
import os
import re

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     'unscaled model', 'gait2392_Oz_NoPsis_wExo_38mk.osim')
EXO_OFFSET = (0.20000000298023224, -0.10450000315904617, 0.0)    # Pelvis2ExoP / ExoP_offset translation
HIP = {'L': ('femur_l', (-0.0707, -0.0661, -0.0835)),            # hip_l pelvis_offset translation
       'R': ('femur_r', (-0.0707, -0.0661, 0.0835))}             # hip_r pelvis_offset translation

s = open(MODEL, encoding='utf-8').read()
for side, (femur, hip) in HIP.items():
    pat = re.compile(r'(<Marker name="%sGTR">.*?<socket_parent_frame>)/bodyset/%s(</socket_parent_frame>'
                     r'.*?<location>)([^<]*)(</location>)' % (side, femur), re.S)
    m = pat.search(s)
    if not m:
        print(f'{side}GTR: not parented to {femur}, skipped')
        continue
    loc = [float(v) for v in m.group(3).split()]
    new = [loc[i] + hip[i] + EXO_OFFSET[i] for i in range(3)]
    print(f'{side}GTR  {femur} {loc} -> ExoP {new}')
    s = pat.sub(lambda mm: mm.group(1) + '/bodyset/ExoP' + mm.group(2)
                + ' '.join(repr(round(v, 12)) for v in new) + mm.group(4), s, count=1)
open(MODEL, 'w', encoding='utf-8', newline='').write(s)
