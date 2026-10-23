# sim2real

OpenSim pipeline: scale model -> inverse kinematics (IK) -> inverse dynamics (ID) -> plots.

## Folder layout
```
sim2real/
├── Data/<MM_DD_YYYY SUBJECT>/       raw Vicon exports, e.g. "10_08_2026 AB01"
│   ├── standing.trc/.mot/.csv       static trial (used for scaling, never run through IK/ID)
│   └── <trial>.trc/.mot/.csv        every other trial is processed
├── Subject Information.txt          mass / height / age per subject (AB01, AB02, ...)
├── unscaled model/gait2392.osim     generic model
├── setup/scale_setup_template.xml   scaling settings (measurements + marker weights)
├── scripts/
│   ├── config.py                    IK marker weights, ID filter, subject lookup
│   ├── scale_model.py               unscaled model + static .trc -> scaled model (paths at the top)
│   ├── run_ik_id.py                 batch IK + ID of a list of trials (paths at the top); scales the
│   │                                model first if it isn't in Results/<session>/model/ yet
│   ├── hip_from_femur.py            scaled model + trial .trc -> GTR rebuilt from LE / ME (.trc with
│   │                                RGTR_est / LGTR_est added), error plots
│   ├── run_body_kinematics.py       OpenSim BodyKinematics (Analyze tool) of every trial + its static
│   ├── util_IKIDGaitPlot.py         IK and ID figures of a list of trials over the gait cycle (settings at the top)
│   ├── util_BodyKinematicsPlot.py   femur vs pelvis / torso and exo vs human thigh, 0 = static pose
│   ├── util_marker_report.py        marker-error tables printed by scale_model.py and run_ik_id.py
│   ├── util_progress_bar.py         progress bar
│   └── util_stl2vtp.py              convert .stl geometry to .vtp
├── Results/<session>/               everything generated (safe to delete)
│   ├── model/<static>_scaled.osim   scaled model (only file kept from scaling)
│   ├── bodykinematics/              figures of util_BodyKinematicsPlot.py
│   └── <trial>/                     IK .mot, ID .sto, marker errors, external loads .xml,
│                                    BodyKinematics pos / vel / acc .sto
└── environment.yml
```

## Usage
```powershell
conda env create -f environment.yml   # once
conda activate OpenSimPyEnv

python scripts/run_ik_id.py        # 1. IK + ID of every trial in TRIALS (scales the model if needed)
python scripts/scale_model.py      #    optional: (re)scale on its own, e.g. after changing the template
python scripts/hip_from_femur.py   #    GTR reconstruction with the scaled model
python scripts/util_IKIDGaitPlot.py   # 2. plots from step 1
python scripts/run_body_kinematics.py      # 3. BodyKinematics of the trials from step 1 (+ statics)
python scripts/util_BodyKinematicsPlot.py  #    segment-vs-segment plots from step 3
```
Each script reads its inputs from the settings at its top and only writes its own outputs.

New session: in `run_ik_id.py` set `SESSION`, `TRIALS` (any number), `STATIC_TRIAL`, `STATIC_TIME_WINDOW`,
`SUBJECT_ID` and the model / scale template / IK setup for the marker protocol. The scaled model is
loaded from `Results/<session>/model/<STATIC_TRIAL>_scaled.osim`; delete it to force a rescale.
Make sure the subject is listed in `Subject Information.txt`.

Raw Vicon data default path: `C:\Users\oscar\Documents\Vicon & Treadmill Data`
