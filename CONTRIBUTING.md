# Contributing

## Build and test

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DAFDTD_BUILD_TESTS=ON [-DAFDTD_BUILD_TOOLS=ON]
cmake --build build -j
ctest --test-dir build -L cpu          # no GPU: API layout, refusals, kernels_generated (needs Python 3)
ctest --test-dir build -L gpu          # one supported GPU; -L gpu2 needs two
pip install .[test] && pytest tests/python
```

`-DAFDTD_BUILD_TOOLS=ON` also builds `tools/bench` (timed runs through the public API, one JSON line per run).

## Rules

- **Bit-exactness.** Every change keeps the state bit-identical to `tests/oracle/reference.cu` for every plan, step
  count, split of the steps, memory mode and GPU count. The GPU tests compare every value; a change that makes any of
  them fail is not accepted.
- **Kernels.** `src/kernels/*.cuh` are generated. Change them only through `tools/gen` and commit the result;
  `python3 tools/gen/gen.py --check` (the `kernels_generated` test) must pass.
- **Performance.** Report speed changes with numbers measured on the same machine before and after (`tools/bench`).
  Never fill in a number that was not measured.
- **API.** `include/alpha_fdtd.h` is API v1: functions, structs and constants do not change meaning or layout
  (`tests/data/api_v1.txt` checks the layout). New features are new functions.
- Every source file starts with `SPDX-License-Identifier: MIT`. Contributions are under the MIT license.

## Reporting bugs and security issues

Open an issue at https://github.com/spinsphotonics/alpha-fdtd/issues. For a wrong result, include the GPU type and
count, the domain size, the step count, the plan description (`describe()` / `afdtd_plan_describe`) and a script that
reproduces it. For a security issue, open an issue asking for a private contact instead of posting the details.
