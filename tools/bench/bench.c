/* SPDX-License-Identifier: MIT
 * bench.c -- timed runs through the public API (developer tool).  Prints one JSON line per run.
 *
 *   afdtd_bench NX NY NZ STEPS [estimate|measure|patient] [NGPUS] [HOST_GIB] [force_host] [repeat=N] [wisdom=FILE]
 *
 * The fields are a fixed lossy problem set by broadcast (E = 2^-4, H = 0, Ca = 0.99, Cb = 0.25): the speed does not depend on the data.
 */
#include <alpha_fdtd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static int die(const char* what) { fprintf(stderr, "afdtd_bench: %s: %s\n", what, afdtd_last_error()); return 1; }

int main(int argc, char** argv)
{
  if (argc < 5) { fprintf(stderr, "usage: afdtd_bench NX NY NZ STEPS [estimate|measure|patient] [NGPUS] [HOST_GIB] [force_host] [repeat=N] [wisdom=FILE]\n"); return 2; }
  const long long nx = atoll(argv[1]), ny = atoll(argv[2]), nz = atoll(argv[3]), steps = atoll(argv[4]);
  afdtd_options o; memset(&o, 0, sizeof o);
  int repeat = 1, nnum = 0; const char* wis = NULL;
  for (int i = 5; i < argc; ++i) {
    if (!strcmp(argv[i], "estimate")) o.rigor = AFDTD_ESTIMATE;
    else if (!strcmp(argv[i], "measure")) o.rigor = AFDTD_MEASURE;
    else if (!strcmp(argv[i], "patient")) o.rigor = AFDTD_PATIENT;
    else if (!strcmp(argv[i], "force_host")) o.flags |= AFDTD_FORCE_HOST;
    else if (!strncmp(argv[i], "repeat=", 7)) repeat = atoi(argv[i] + 7);
    else if (!strncmp(argv[i], "wisdom=", 7)) wis = argv[i] + 7;
    else if (nnum++ == 0) o.ngpus = atoi(argv[i]);
    else o.host_bytes = (size_t)(atof(argv[i]) * 1073741824.0);
  }
  if (wis) { afdtd_import_wisdom_from_filename(wis); if (o.rigor != AFDTD_ESTIMATE) o.rigor = AFDTD_WISDOM_ONLY; }
  const double t0 = now();
  afdtd_plan p = afdtd_plan_create(nx, ny, nz, &o);
  if (!p) return die("plan");
  const double tplan = now() - t0;
  float e = 0.0625f, ca = 0.99f, cb = 0.25f;
  afdtd_box b; memset(&b, 0, sizeof b);
  b.shape[0] = nx; b.shape[1] = ny; b.shape[2] = nz; b.flags = AFDTD_BOX_STRIDED;
  for (int a = 0; a < 3; ++a) {
    b.array[AFDTD_EX + a].data = &e; b.array[AFDTD_EX + a].dtype = AFDTD_F32;
    b.array[AFDTD_CAX + a].data = &ca; b.array[AFDTD_CAX + a].dtype = AFDTD_F32;
    b.array[AFDTD_CBX + a].data = &cb; b.array[AFDTD_CBX + a].dtype = AFDTD_F32;
  }
  const double t1 = now();
  if (afdtd_set(p, &b)) return die("set");
  const double tset = now() - t1;
  for (int r = 0; r < repeat; ++r) {
    const double t2 = now();
    if (afdtd_execute(p, steps)) return die("execute");
    const double sec = now() - t2;
    char d[1024]; afdtd_plan_describe(p, d, sizeof d);
    printf("{\"nx\": %lld, \"ny\": %lld, \"nz\": %lld, \"steps\": %lld, \"run\": %d, \"seconds\": %.4f, \"rate\": %.5e, \"plan_s\": %.2f, \"set_s\": %.2f, \"describe\": \"%s\"}\n",
           nx, ny, nz, steps, r, sec, sec > 0 ? (double)nx * ny * nz * steps / sec : 0.0, tplan, tset, d);
    fflush(stdout);
  }
  if (wis && o.rigor != AFDTD_WISDOM_ONLY) afdtd_export_wisdom_to_filename(wis);
  afdtd_plan_destroy(p);
  return 0;
}
