//===- Utils.h - Utils for Dist dialect  -----------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the utils for the dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _DIST_UTILS_H_INCLUDED_
#define _DIST_UTILS_H_INCLUDED_

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Utils/PassUtils.h>

#include <vector>

namespace imex {
namespace dist {

// *******************************
// ***** Some helper functions ***
// *******************************

// Create a DistTensor from a PTensor and meta data
inline ::mlir::Value
createDistTensor(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                 ::mlir::ValueRange gshape, ::mlir::Value pt,
                 ::mlir::ValueRange loffsets, ::mlir::Value team) {
  return builder.create<::imex::dist::InitDistTensorOp>(loc, pt, team, gshape,
                                                        loffsets);
}

// create operation returning global shape of DistTensor
inline ::mlir::ValueRange createGlobalShapeOf(::mlir::Location &loc,
                                              ::mlir::OpBuilder &builder,
                                              ::mlir::Value dt) {
  return builder.create<::imex::dist::GlobalShapeOfOp>(loc, dt).getGShape();
}

// create operation returning local offsets of DistTensor
inline ::mlir::ValueRange createLocalOffsetsOf(::mlir::Location &loc,
                                               ::mlir::OpBuilder &builder,
                                               ::mlir::Value dt) {
  return builder.create<::imex::dist::LocalOffsetsOfOp>(loc, dt).getLOffsets();
}

// create operation returning the local Tensor of DistTensor
inline ::mlir::Value createLocalTensorOf(::mlir::Location &loc,
                                         ::mlir::OpBuilder &builder,
                                         ::mlir::Value dt) {
  return builder.create<::imex::dist::LocalTensorOfOp>(loc, dt).getLTensor();
}

// create operation returning the team of DistTensor
inline ::mlir::Value createTeamOf(::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Value dt) {
  return builder.create<::imex::dist::TeamOfOp>(loc, dt).getTeam();
}

// create operation returning the re-balanced tensor
inline ::mlir::Value createReBalance(::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Value dt) {
  return builder.create<::imex::dist::ReBalanceOp>(loc, dt);
}

} // namespace dist
} // namespace imex

#endif // _DIST_UTILS_H_INCLUDED_
