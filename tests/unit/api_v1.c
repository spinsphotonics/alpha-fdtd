/* SPDX-License-Identifier: MIT
 * api_v1.c -- API v1 freeze: struct layouts and constant values of alpha_fdtd.h, compared with tests/data/api_v1.txt.
 * Compiled as strict C99 (and, through api_v1_cxx.cpp, as C++17, which also checks every function's exact signature).
 *   afdtd_api_v1 SNAPSHOT        exit 0 when the printed snapshot equals the file;  afdtd_api_v1 -  prints it
 */
#include <alpha_fdtd.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define L(fmt, v) n += snprintf(buf + n, sizeof buf - n, fmt "\n", (long long)(v))
#define S(t) L(#t " size %lld", sizeof(t))
#define O(t, m) L(#t "." #m " offset %lld", offsetof(t, m))
#define K(c) L(#c " %lld", c)

int main(int argc, char** argv)
{
  static char buf[8192], ref[8192];
  int n = 0;
  static afdtd_options zo; static afdtd_box zb; static afdtd_info zi;
  (void)zo; (void)zb; (void)zi;
  K(AFDTD_API_VERSION); K(AFDTD_VERSION_MAJOR); K(AFDTD_VERSION_MINOR); K(AFDTD_VERSION_PATCH);
  K(AFDTD_OK); K(AFDTD_EINVAL); K(AFDTD_ENOMEM); K(AFDTD_ENOPLAN); K(AFDTD_EUNSUPPORTED); K(AFDTD_EDEVICE); K(AFDTD_EWALLS);
  K(AFDTD_EIO); K(AFDTD_EINTERNAL);
  K(AFDTD_ESTIMATE); K(AFDTD_MEASURE); K(AFDTD_PATIENT); K(AFDTD_WISDOM_ONLY); K(AFDTD_FORCE_HOST); K(AFDTD_STRICT_WALLS);
  K(AFDTD_EX); K(AFDTD_EY); K(AFDTD_EZ); K(AFDTD_HX); K(AFDTD_HY); K(AFDTD_HZ); K(AFDTD_CAX); K(AFDTD_CAY); K(AFDTD_CAZ);
  K(AFDTD_CBX); K(AFDTD_CBY); K(AFDTD_CBZ); K(AFDTD_NCOMP); K(AFDTD_BF16); K(AFDTD_F32); K(AFDTD_BOX_STRIDED);
  S(afdtd_array); O(afdtd_array, data); O(afdtd_array, dtype); O(afdtd_array, stride);
  S(afdtd_box); O(afdtd_box, offset); O(afdtd_box, shape); O(afdtd_box, flags); O(afdtd_box, array);
  S(afdtd_options); O(afdtd_options, ngpus); O(afdtd_options, devices); O(afdtd_options, host_bytes); O(afdtd_options, rigor);
  O(afdtd_options, flags);
  S(afdtd_info); O(afdtd_info, nx); O(afdtd_info, ny); O(afdtd_info, nz); O(afdtd_info, ngpus); O(afdtd_info, devices);
  O(afdtd_info, on_host); O(afdtd_info, rigor); O(afdtd_info, from_wisdom); O(afdtd_info, io_block_y);
  O(afdtd_info, gpu_bytes_used); O(afdtd_info, host_bytes_used); O(afdtd_info, steps_done); O(afdtd_info, tail_steps);
  O(afdtd_info, walls_zeroed_coef); O(afdtd_info, walls_zeroed_field); O(afdtd_info, predicted_rate);
  O(afdtd_info, measured_rate); O(afdtd_info, last_rate);
  if (argc < 2 || !strcmp(argv[1], "-")) { fputs(buf, stdout); return 0; }
  {
    FILE* f = fopen(argv[1], "rb");
    size_t m;
    if (!f) { fprintf(stderr, "cannot read %s\n", argv[1]); return 1; }
    m = fread(ref, 1, sizeof ref - 1, f); ref[m] = 0; fclose(f);
  }
  if (strcmp(buf, ref)) { printf("API v1 snapshot differs from %s; this build prints:\n%s", argv[1], buf); return 1; }
  printf("{\"test\": \"api_v1\", \"passed\": true}\n");
  return 0;
}
