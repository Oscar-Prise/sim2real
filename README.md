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
│   ├── run_ik_id.py                 scale + IK + ID for every trial in a session (paths at the top)
│   ├── plot_ik_id.py                plot one trial over time and per gait cycle (files + options at the top)
│   ├── scale_model.py               scaling with the OpenSim API
│   └── progress.py                  progress bar
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

python scripts/run_ik_id.py                          # data paths set at the top of that script
python scripts/plot_ik_id.py                         # files set at the top of that script
```

New session: set `DATA_DIR`, `RESULTS_DIR`, `STATIC_TRC`, `STATIC_TIME_WINDOW` and `SUBJECT_ID`
at the top of `run_ik_id.py`, and make sure the subject is listed in `Subject Information.txt`.

Raw Vicon data default path: `C:\Users\oscar\Documents\Vicon & Treadmill Data`
