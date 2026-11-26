// SPDX-License-Identifier: MIT
// api_v1_cxx.cpp -- api_v1.c compiled as C++17, plus the exact signature of every function of API v1.
#include "api_v1.c"
#include <type_traits>
#define SIG(f, ...) static_assert(std::is_same<decltype(f), __VA_ARGS__>::value, #f " changed");
SIG(afdtd_plan_create, afdtd_plan(int64_t, int64_t, int64_t, const afdtd_options*))
SIG(afdtd_plan_destroy, void(afdtd_plan))
SIG(afdtd_bytes_needed, int(int64_t, int64_t, int64_t, int, size_t*, size_t*))
SIG(afdtd_set, int(afdtd_plan, const afdtd_box*))
SIG(afdtd_get, int(afdtd_plan, const afdtd_box*))
SIG(afdtd_execute, int(afdtd_plan, int64_t))
SIG(afdtd_checksums, int(afdtd_plan, uint64_t*))
SIG(afdtd_plan_info, int(afdtd_plan, afdtd_info*))
SIG(afdtd_plan_describe, int(afdtd_plan, char*, size_t))
SIG(afdtd_export_wisdom_to_string, char*(void))
SIG(afdtd_import_wisdom_from_string, int(const char*))
SIG(afdtd_export_wisdom_to_filename, int(const char*))
SIG(afdtd_import_wisdom_from_filename, int(const char*))
SIG(afdtd_forget_wisdom, void(void))
SIG(afdtd_free, void(void*))
SIG(afdtd_version, const char*(void))
SIG(afdtd_last_error, const char*(void))
