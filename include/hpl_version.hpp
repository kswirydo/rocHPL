/* ---------------------------------------------------------------------
 * -- High Performance Computing Linpack Benchmark (HPL)
 *    Noel Chalmers
 *    (C) 2018-2025 Advanced Micro Devices, Inc.
 *    See the rocHPL/LICENCE file for details.
 *
 *    SPDX-License-Identifier: (BSD-3-Clause)
 * ---------------------------------------------------------------------
 */

#ifndef HPL_VERSION_HPP
#define HPL_VERSION_HPP

// clang-format off
#define __ROCHPL_VER_MAJOR     7
#define __ROCHPL_VER_MINOR     0
#define __ROCHPL_VER_PATCH     1
#define __ROCHPL_VER_TWEAK     daed9ea-dirty
// clang-format on

#define __ROCHPL_VER \
  10000 * __ROCHPL_VER_MAJOR + 100 * __ROCHPL_VER_MINOR + __ROCHPL_VER_PATCH

#endif // VERSION_HPP
