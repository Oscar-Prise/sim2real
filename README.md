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
│   ├── run_ik_id.py                 scaled model + trial .trc / .mot -> IK + ID (paths at the top)
│   ├── hip_from_femur.py            scaled model + trial .trc -> GTR rebuilt from LE / ME (.trc with
│   │                                RGTR_est / LGTR_est added), error plots
│   ├── plot_ik_id.py                plot one trial over time and per gait cycle (files + options at the top)
│   ├── util_marker_report.py        marker-error tables printed by scale_model.py and run_ik_id.py
│   ├── util_progress_bar.py         progress bar
│   └── util_stl2vtp.py              convert .stl geometry to .vtp
├── Results/<session>/               everything generated (safe to delete)
│   ├── model/standing_scaled.osim   final scaled model (the one to use)
│   ├── model/intermediate/          in-between scaling outputs (for debugging only)
│   └── <trial>/                     IK .mot, ID .sto, marker errors, external loads .xml
└── environment.yml
```

## Usage
```powershell
conda env create -f environment.yml   # once
conda activate OpenSimPyEnv

python scripts/scale_model.py      # 1. once per session: makes the scaled model
python scripts/run_ik_id.py        # 2. IK + ID with that model
python scripts/hip_from_femur.py   #    GTR reconstruction with that model (needs step 1 only)
python scripts/plot_ik_id.py       # 3. plots from step 2
```
Each script reads its inputs from the settings at its top and only writes its own outputs.

New session: in `scale_model.py` set `STATIC_TRC`, `STATIC_TIME_WINDOW`, `SUBJECT_ID`, `OUTPUT_MODEL`
(and the model / template for the marker protocol), then point `SCALED_MODEL` in `run_ik_id.py` and
`hip_from_femur.py` to that output. Make sure the subject is listed in `Subject Information.txt`.

Raw Vicon data default path: `C:\Users\oscar\Documents\Vicon & Treadmill Data`
