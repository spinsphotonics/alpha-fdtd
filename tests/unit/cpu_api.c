/* SPDX-License-Identifier: MIT
 * cpu_api.c -- the C API's refusals, error text, memory sizes and wisdom, through libalphafdtd; needs no GPU.
 *   afdtd_cpu_api TMPDIR        prints one JSON line; exit 0 when every check passed
 */
#include <alpha_fdtd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fails = 0, checks = 0;
static void check(int ok, const char* what)
{
  ++checks;
  if (!ok) { ++fails; fprintf(stderr, "FAILED: %s (last error: %s)\n", what, afdtd_last_error()); }
}
static int starts(const char* s, const char* p) { return !strncmp(s, p, strlen(p)); }
static int refused(int64_t nx, int64_t ny, int64_t nz, int ngpus, const int* devs, int rigor, unsigned flags)
{
  afdtd_options o;
  afdtd_plan p;
  memset(&o, 0, sizeof o);
  o.ngpus = ngpus; o.devices = devs; o.rigor = rigor; o.flags = flags;
  p = afdtd_plan_create(nx, ny, nz, &o);
  if (p) { afdtd_plan_destroy(p); return 0; }
  return starts(afdtd_last_error(), "AFDTD_EINVAL: ");
}
static char* wisdom(void) { return afdtd_export_wisdom_to_string(); }

int main(int argc, char** argv)
{
  const char* tmp = argc > 1 ? argv[1] : ".";
  char path[4096], *w0, *w1, *w2;
  size_t g1 = 0, h1 = 0, g2 = 0, h2 = 0;
  int dev1[1] = {0};
  afdtd_box box;
  afdtd_info info;
  uint64_t cs[6];
  char buf[8];
  const char* rec_e = "kset=000000000000 gpu=X ngpus=1 n=16x16x16 mode=gpu rigor=estimate rate=1 sched=\"a\"";
  const char* rec_m = "kset=000000000000 gpu=X ngpus=1 n=16x16x16 mode=gpu rigor=measure rate=2 sched=\"b\"";
  char text[1024];

  /* version, error text */
  check(!strcmp(afdtd_version(), "0.2.0"), "afdtd_version");
  check(AFDTD_API_VERSION == 2, "AFDTD_API_VERSION");
  check(afdtd_bytes_needed(16, 16, 16, 1, &g1, &h1) == AFDTD_OK && !strcmp(afdtd_last_error(), ""), "error text empty after a success");

  /* memory sizes */
  check(afdtd_bytes_needed(1000, 1000, 1000, 1, &g1, &h1) == AFDTD_OK && g1 >= (size_t)24e9 && h1 >= (size_t)24e9, "bytes_needed >= 24 B per cell");
  check(afdtd_bytes_needed(1000, 1000, 1000, 2, &g2, &h2) == AFDTD_OK && g2 < g1, "bytes_needed per GPU falls with 2 GPUs");
  check(afdtd_bytes_needed(1000, 1000, 1000, 1, NULL, NULL) == AFDTD_OK, "bytes_needed with NULL outputs");
  check(afdtd_bytes_needed(15, 16, 16, 1, &g1, &h1) == AFDTD_EINVAL && starts(afdtd_last_error(), "AFDTD_EINVAL: "), "bytes_needed refuses 15 cells");
  check(afdtd_bytes_needed(16, 16, 16, 9, &g1, &h1) == AFDTD_EINVAL, "bytes_needed refuses 9 GPUs");
  check(afdtd_bytes_needed(1 << 20, 1 << 20, 1 << 20, 1, &g1, &h1) == AFDTD_EINVAL, "bytes_needed refuses sizes that would overflow");

  /* plan options refused before any GPU is touched */
  check(refused(15, 16, 16, 1, NULL, AFDTD_ESTIMATE, 0), "size 15 refused");
  check(refused(16, 16, (1 << 20) + 1, 1, NULL, AFDTD_ESTIMATE, 0), "size above 2^20 refused");
  check(refused(16, 16, 16, 1, NULL, 4, 0), "rigor 4 refused");
  check(refused(16, 16, 16, 1, NULL, -1, 0), "rigor -1 refused");
  check(refused(16, 16, 16, 1, NULL, AFDTD_ESTIMATE, AFDTD_MEASURE), "a rigor value in flags refused");
  check(refused(16, 16, 16, 1, NULL, (int)AFDTD_FORCE_HOST, 0), "a flag in rigor refused");
  check(refused(16, 16, 16, 9, NULL, AFDTD_ESTIMATE, 0), "ngpus 9 refused");
  check(refused(16, 16, 16, -1, NULL, AFDTD_ESTIMATE, 0), "ngpus -1 refused");
  check(refused(16, 16, 16, 0, dev1, AFDTD_ESTIMATE, 0), "devices with ngpus 0 refused");

  /* NULL plan and NULL arguments */
  memset(&box, 0, sizeof box);
  check(afdtd_set(NULL, &box) == AFDTD_EINVAL && afdtd_get(NULL, &box) == AFDTD_EINVAL, "set/get on NULL plan");
  check(afdtd_execute(NULL, 7) == AFDTD_EINVAL && afdtd_checksums(NULL, cs) == AFDTD_EINVAL, "execute/checksums on NULL plan");
  check(afdtd_plan_info(NULL, &info) == AFDTD_EINVAL && afdtd_plan_describe(NULL, buf, sizeof buf) == AFDTD_EINVAL, "info/describe on NULL plan");
  afdtd_plan_destroy(NULL);
  check(!strcmp(afdtd_last_error(), ""), "afdtd_plan_destroy(NULL) clears the error text");
  check(afdtd_import_wisdom_from_string(NULL) == AFDTD_EINVAL && afdtd_import_wisdom_from_filename(NULL) == AFDTD_EINVAL, "NULL wisdom arguments");

  /* wisdom */
  afdtd_forget_wisdom();
  w0 = wisdom();
  check(w0 && !strcmp(w0, "alpha-fdtd-wisdom 1\n"), "empty wisdom is the header line");
  check(afdtd_import_wisdom_from_string("") == AFDTD_EINVAL, "empty text refused");
  check(afdtd_import_wisdom_from_string("fftw-wisdom\n") == AFDTD_EINVAL, "wrong header refused");
  snprintf(text, sizeof text, "alpha-fdtd-wisdom 1\n%s\nnot a record\n", rec_m);
  check(afdtd_import_wisdom_from_string(text) == AFDTD_EINVAL, "a bad line refused");
  snprintf(text, sizeof text, "alpha-fdtd-wisdom 1\n%.*s\n", (int)strlen(rec_m) - 1, rec_m);
  check(afdtd_import_wisdom_from_string(text) == AFDTD_EINVAL, "an unterminated schedule refused");
  w1 = wisdom();
  check(w1 && !strcmp(w1, w0), "nothing merged from a text with an error");
  snprintf(text, sizeof text, "alpha-fdtd-wisdom 1\n%s\n\n%s\n", rec_e, rec_m);
  check(afdtd_import_wisdom_from_string(text) == 2, "two records read");
  snprintf(text, sizeof text, "alpha-fdtd-wisdom 1\r\n%s\r\n", rec_e);
  check(afdtd_import_wisdom_from_string(text) == 1, "CRLF text read");
  afdtd_free(w1);
  w1 = wisdom();
  snprintf(text, sizeof text, "alpha-fdtd-wisdom 1\n%s\n", rec_m);
  check(w1 && !strcmp(w1, text), "higher rigor kept, unknown kernel set kept");
  snprintf(path, sizeof path, "%s/afdtd_cpu_api.wisdom", tmp);
  check(afdtd_export_wisdom_to_filename(path) == AFDTD_OK, "export to a file");
  afdtd_forget_wisdom();
  check(afdtd_import_wisdom_from_filename(path) == 1, "import from the file");
  w2 = wisdom();
  check(w2 && !strcmp(w1, w2), "file round trip");
  remove(path);
  check(afdtd_import_wisdom_from_filename(path) == AFDTD_EIO && starts(afdtd_last_error(), "AFDTD_EIO: "), "missing file is EIO");
  snprintf(path, sizeof path, "%s/no-such-directory/w", tmp);
  check(afdtd_export_wisdom_to_filename(path) == AFDTD_EIO, "unwritable file is EIO");
  afdtd_forget_wisdom();
  afdtd_free(w0); afdtd_free(w1); afdtd_free(w2);

  /* with no supported GPU, a valid plan is refused as unsupported; with one, it is created */
  {
    afdtd_plan p = afdtd_plan_create(16, 16, 16, NULL);
    if (p) afdtd_plan_destroy(p);
    else check(starts(afdtd_last_error(), "AFDTD_EUNSUPPORTED: "), "a valid plan without a GPU is EUNSUPPORTED");
  }
  printf("{\"test\": \"cpu_api\", \"passed\": %s, \"checks\": %d, \"failed\": %d}\n", fails ? "false" : "true", checks, fails);
  return fails ? 1 : 0;
}
