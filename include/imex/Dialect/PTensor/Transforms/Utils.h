//===-- PasUtils.h - PTensor utils ------------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This header file defines utils for PTensor passes.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_UTILS_H_INCLUDED_
#define _PTENSOR_UTILS_H_INCLUDED_

#include <imex/Utils/ArithUtils.h>

namespace imex {

/// create operations computing the space between elements a
/// linspace(start, stop, num) would have.
/// @return step a linspace(start, stop, num) would have (::mlir::Value)
inline ::mlir::Value createStepLinSpace(::mlir::OpBuilder &builder,
                                        ::mlir::Location loc,
                                        ::mlir::Value start, ::mlir::Value stop,
                                        ::mlir::Value num, bool endpoint) {
  auto typ = builder.getF64Type();
  start = createCast(loc, builder, start, typ);
  stop = createCast(loc, builder, stop, typ);
  num = createCast(loc, builder, num, typ);
  if (endpoint) {
    auto one = createFloat(loc, builder, 1);
    num = builder.create<::mlir::arith::SubFOp>(loc, num, one);
  }
  return builder.create<::mlir::arith::DivFOp>(
      loc, builder.create<::mlir::arith::SubFOp>(loc, stop, start), num);
}

} // namespace imex

#endif // _PTENSOR_UTILS_H_INCLUDED_
