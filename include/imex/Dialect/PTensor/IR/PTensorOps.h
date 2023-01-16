//===- PTensorOps.h - PTensor dialect  -------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensor_OPS_H_INCLUDED_
#define _PTensor_OPS_H_INCLUDED_

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>

#include "PTensorDefs.h"
#include <imex/Dialect/PTensor/Utils/Utils.h>

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.h.inc>

#endif // _PTensor_OPS_H_INCLUDED_
