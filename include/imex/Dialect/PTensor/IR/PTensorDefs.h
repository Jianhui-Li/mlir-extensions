//===- PTensorOps.h - PTensor dialect  -------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the PTensor dialect's enums and other defs.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensor_DEFS_H_INCLUDED_
#define _PTensor_DEFS_H_INCLUDED_

namespace imex {
namespace ptensor {

enum DType : int8_t { F64, F32, I64, U64, I32, U32, I16, U16, I8, U8, I1 };

/// The set of supported elementwise binary operations
enum EWBinOpId : int {
  ADD,
  AND,
  ATAN2,
  BITWISE_AND,
  BITWISE_LEFT_SHIFT,
  BITWISE_OR,
  BITWISE_RIGHT_SHIFT,
  BITWISE_XOR,
  EQUAL,
  FLOOR_DIVIDE,
  GREATER,
  GREATER_EQUAL,
  LESS,
  LESS_EQUAL,
  LOGADDEXP,
  LOGICAL_AND,
  LOGICAL_OR,
  LOGICAL_XOR,
  LSHIFT,
  MATMUL,
  MAXIMUM,
  MINIMUM,
  MODULO,
  MULTIPLY,
  NOT_EQUAL,
  OR,
  POWER,
  SUBTRACT,
  TRUE_DIVIDE,
  XOR,
  EWBINOPID_LAST
};

/// The set of supported reduction operations
enum ReduceOpId : int { MAX, MEAN, MIN, PROD, SUM, STD, VAR, REDUCEOPID_LAST };

} // namespace ptensor
} // namespace imex

#endif // _PTensor_DEFS_H_INCLUDED_
