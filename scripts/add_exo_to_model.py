"""
Make the exo version of a human-only unscaled model (e.g. a subject-specific one): the exo bodies,
their joints and the exo markers are copied from an existing exo model, and the human markers
the exo covers (not in the exo trials) are dropped. Set the paths below, then run:

    python scripts/add_exo_to_model.py

The human part (bodies, joints, human markers) is left exactly as in HUMAN_MODEL.
"""
import copy
import os
import xml.etree.ElementTree as ET

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(_ROOT, 'Data', '10_22_2026 AB02', 'Model and scale')
HUMAN_MODEL = os.path.join(MODEL_DIR, 'maria.osim')
OUTPUT_MODEL = os.path.join(MODEL_DIR, 'maria_Exo.osim')
EXO_SOURCE_MODEL = os.path.join(_ROOT, 'unscaled model', 'gait2392_Exo_Clust_n_Psis.osim')
EXO_BODIES = ('ExoP', 'ExoTR', 'ExoTL')
DROP_MARKERS = ('LGTR', 'RGTR')   # under the exo belt


def _on_exo_body(element):
    """True if any frame / socket of `element` is attached to an exo body."""
    return any(e.text and e.text.strip().split('/')[-1] in EXO_BODIES
               for e in element.iter() if e.tag.startswith('socket_parent'))


def add_exo(human_model, exo_source_model, output_model):
    tree = ET.parse(human_model)
    human = tree.getroot().find('Model')
    exo = ET.parse(exo_source_model).getroot().find('Model')

    bodies = human.find('BodySet/objects')
    names = {b.get('name') for b in bodies}
    for body in exo.find('BodySet/objects'):
        if body.get('name') in EXO_BODIES and body.get('name') not in names:
            bodies.append(copy.deepcopy(body))

    joints = human.find('JointSet/objects')
    names = {j.get('name') for j in joints}
    for joint in exo.find('JointSet/objects'):
        if _on_exo_body(joint) and joint.get('name') not in names:
            joints.append(copy.deepcopy(joint))

    markers = human.find('MarkerSet/objects')
    for marker in [m for m in markers if m.get('name') in DROP_MARKERS]:
        markers.remove(marker)
    names = {m.get('name') for m in markers}
    for marker in exo.find('MarkerSet/objects'):
        if _on_exo_body(marker) and marker.get('name') not in names:
            markers.append(copy.deepcopy(marker))

    human.set('name', human.get('name') + '_Exo')
    tree.write(output_model, encoding='UTF-8', xml_declaration=True)
    print(f"{os.path.basename(output_model)}: {len(bodies)} bodies, {len(joints)} joints, {len(markers)} markers "
          f"(exo from {os.path.basename(exo_source_model)}, dropped {', '.join(DROP_MARKERS)})")
    return output_model


if __name__ == '__main__':
    add_exo(HUMAN_MODEL, EXO_SOURCE_MODEL, OUTPUT_MODEL)
