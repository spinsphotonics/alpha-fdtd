/* SPDX-License-Identifier: MIT
 * quickstart.c -- the C quick start: a 128^3 vacuum box, an Ez pulse, broadcast coefficients, 700 steps.
 *   cc -std=c99 quickstart.c -I<prefix>/include -L<prefix>/lib -lalphafdtd -lm -o quickstart
 */
#include <alpha_fdtd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void)
{
  enum { N = 128 };                                          /* 128^3 cells, about 50 MB of state */
  const int64_t n = (int64_t)N * N * N, s[3] = {N * N, N, 1}, mid = (N / 2 * N + N / 2) * N + N / 2;
  float *ez = malloc(n * sizeof *ez), one = 1.0f, cb = 0.25f;   /* vacuum, dt = 1/2: Cb' = dt * dt / eps */
  if (!ez) return 1;
  for (int64_t c = 0; c < n; ++c) {                          /* Ez: a Gaussian in x-y, uniform in z (div E = 0) */
    const float x = (float)(c / (N * N) - N / 2), y = (float)(c / N % N - N / 2);
    ez[c] = expf(-(x * x + y * y) / 32.0f);
  }
  afdtd_import_wisdom_from_filename("my.wisdom");            /* optional: a missing file is fine */
  afdtd_options opt = { .rigor = AFDTD_ESTIMATE, .flags = AFDTD_STRICT_WALLS };   /* 1 GPU, GPU memory */
  afdtd_plan p = afdtd_plan_create(N, N, N, &opt);
  if (!p) { fprintf(stderr, "alpha-fdtd: %s\n", afdtd_last_error()); return 1; }
  afdtd_box in = { .shape = {N, N, N}, .flags = AFDTD_BOX_STRIDED };   /* H stays +0: its arrays are NULL */
  in.array[AFDTD_EZ] = (afdtd_array){ .data = ez, .dtype = AFDTD_F32, .stride = {s[0], s[1], s[2]} };
  for (int a = 0; a < 3; ++a) {                              /* zero strides: one value broadcast to the box */
    in.array[AFDTD_CAX + a] = (afdtd_array){ .data = &one, .dtype = AFDTD_F32 };
    in.array[AFDTD_CBX + a] = (afdtd_array){ .data = &cb, .dtype = AFDTD_F32 };
  }
  afdtd_box out = { .shape = {N, N, N} };                    /* dense C order */
  out.array[AFDTD_EZ] = (afdtd_array){ .data = ez, .dtype = AFDTD_F32 };
  for (int frame = 0; frame < 10; ++frame) {
    if ((frame == 0 && afdtd_set(p, &in)) || afdtd_execute(p, 70) || afdtd_get(p, &out)) {
      fprintf(stderr, "alpha-fdtd: %s\n", afdtd_last_error()); return 1;
    }
    printf("step %d: Ez(centre) = %g\n", 70 * (frame + 1), ez[mid]);
  }
  afdtd_export_wisdom_to_filename("my.wisdom");
  afdtd_plan_destroy(p);
  free(ez);
  return 0;
}
