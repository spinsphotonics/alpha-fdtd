# SPDX-License-Identifier: MIT
"""Plan by timing once (rigor="measure"), keep the result as wisdom, and reuse it in later runs without timing."""
import os
import sys
import alpha_fdtd as af

path = sys.argv[1] if len(sys.argv) > 1 else "alpha-fdtd.wisdom"
shape = (256, 256, 256)

if os.path.exists(path):
    print("records read:", af.import_wisdom_file(path))
with af.Plan(shape, rigor="measure") as plan:        # times candidates the first time; instant with a record
    i = plan.info
    print("from wisdom:", i.from_wisdom, " rigor:", i.rigor, f" measured rate: {i.measured_rate:.3g} cells/s")
af.export_wisdom_file(path)

# A program that must never spend time planning asks for the record and fails without it (NoPlan):
with af.Plan(shape, rigor="wisdom_only") as plan:
    print(plan.describe())
