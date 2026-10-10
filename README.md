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
│   ├── config.py                    defaults: session to run/plot, static window, IK weights
│   ├── run_ik_id.py                 scale + IK + ID for every trial in a session
│   ├── plot_ik_id.py                load + plot one trial
│   ├── scale_model.py               scaling with the OpenSim API
│   └── progress.py                  progress bar
├── Results/<session>/               everything generated (safe to delete)
│   ├── model/                       scaled model
│   └── <trial>/                     IK .mot, ID .sto, marker errors, external loads .xml
└── environment.yml
```

## Usage
```powershell
conda env create -f environment.yml   # once
conda activate OpenSimPyEnv

python scripts/run_ik_id.py                          # session set in config.py
python scripts/run_ik_id.py "09_21_2026 AB02"        # a specific session
python scripts/run_ik_id.py --all                    # every session in Data/
python scripts/run_ik_id.py "10_08_2026 AB01" --trials lg_1p2_walking

python scripts/plot_ik_id.py "10_08_2026 AB01" lg_1p2_walking
```

New session: drop the folder into `Data/` named `MM_DD_YYYY SUBJECT`, name the static trial
`standing`, and make sure the subject is listed in `Subject Information.txt`.

Raw Vicon data default path: `C:\Users\oscar\Documents\Vicon & Treadmill Data`
