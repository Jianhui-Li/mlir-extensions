//===- Utils.h - Utils for Dist dialect  -----------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
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
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Dominance.h>

#include <vector>

namespace imex {
namespace dist {

// *******************************
// ***** Some helper functions ***
// *******************************

// Create a DistTensor from a PTensor and meta data
inline ::mlir::Value
createDistTensor(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                 ::mlir::Value pt, bool balanced, ::mlir::ValueRange gshape,
                 ::mlir::ValueRange loffsets, ::mlir::Value team) {
  return builder.create<::imex::dist::InitDistTensorOp>(loc, pt, team, balanced,
                                                        gshape, loffsets);
}

// create operation returning global shape of DistTensor
inline ::mlir::ValueRange createGlobalShapeOf(const ::mlir::Location &loc,
                                              ::mlir::OpBuilder &builder,
                                              ::mlir::Value dt) {
  return builder.create<::imex::dist::GlobalShapeOfOp>(loc, dt).getGShape();
}

// create operation returning local offsets of DistTensor
inline ::mlir::ValueRange createLocalOffsetsOf(const ::mlir::Location &loc,
                                               ::mlir::OpBuilder &builder,
                                               ::mlir::Value dt) {
  return builder.create<::imex::dist::LocalOffsetsOfOp>(loc, dt).getLOffsets();
}

// create operation returning the local Tensor of DistTensor
inline ::mlir::Value createLocalTensorOf(const ::mlir::Location &loc,
                                         ::mlir::OpBuilder &builder,
                                         ::mlir::Value dt) {
  return builder.create<::imex::dist::LocalTensorOfOp>(loc, dt).getLTensor();
}

// create operation returning the team of DistTensor
inline ::mlir::Value createTeamOf(const ::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Value dt) {
  return builder.create<::imex::dist::TeamOfOp>(loc, dt).getTeam();
}

inline ::mlir::Value createNProcs(const ::mlir::Location &loc,
                                  ::mlir::OpBuilder &builder,
                                  ::mlir::Value team) {
  return builder.create<::imex::dist::NProcsOp>(loc, team);
}

inline ::mlir::Value createPRank(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::Value team) {
  return builder.create<::imex::dist::PRankOp>(loc, team);
}

// create operation returning balanced status of DistTensor
inline ::mlir::Value createIsBalanced(const ::mlir::Location &loc,
                                      ::mlir::OpBuilder &builder,
                                      ::mlir::Value dt) {
  return builder.create<::imex::dist::IsBalancedOp>(loc, dt).getIsBalanced();
}

// create operation returning the re-partitioned tensor
inline ::mlir::Value createRePartition(const ::mlir::Location &loc,
                                       ::mlir::OpBuilder &builder,
                                       ::mlir::Value dt,
                                       const ::mlir::ValueRange &offs,
                                       const ::mlir::ValueRange &szs) {
  return builder.create<::imex::dist::RePartitionOp>(loc, dt.getType(), dt,
                                                     offs, szs);
}
// create operation returning the re-partitioned tensor
inline ::imex::dist::RePartitionOp
createRePartition(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                  ::mlir::Value dt) {
  return builder.create<::imex::dist::RePartitionOp>(loc, dt);
}

inline auto createLocalPartition(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, ::mlir::Value dt,
                                 ::mlir::Value team = {},
                                 ::mlir::ValueRange gShape = {}) {
  if (!team)
    team = createTeamOf(loc, builder, dt);
  if (gShape.empty())
    gShape = createGlobalShapeOf(loc, builder, dt);
  auto nProcs = createNProcs(loc, builder, team);
  auto pRank = createPRank(loc, builder, team);
  return builder.create<::imex::dist::LocalPartitionOp>(loc, nProcs, pRank,
                                                        gShape);
}

// return true if there is a write operation between a and b (including b) to
// any operand of b returns false if a does not properly dominate b or no write
// in between a and b
inline bool hasWriteBetween(::mlir::Operation *a, ::mlir::Operation *b,
                            ::mlir::DominanceInfo &dom) {
  if (!dom.properlyDominates(a, b))
    return false;
  if (::mlir::dyn_cast<::imex::ptensor::InsertSliceOp>(b) ||
      ::mlir::dyn_cast<::imex::ptensor::CreateOp>(b) ||
      ::mlir::dyn_cast<::imex::ptensor::LinSpaceOp>(b) ||
      ::mlir::dyn_cast<::imex::ptensor::ReductionOp>(b)) {
    return true;
  } else if (auto op = ::mlir::dyn_cast<::imex::dist::InitDistTensorOp>(b)) {
    auto dop = op.getPTensor().getDefiningOp();
    return dop && hasWriteBetween(a, dop, dom);
  } else if (auto op = ::mlir::dyn_cast<::imex::ptensor::EWBinOp>(b)) {
    auto lhs = op.getLhs().getDefiningOp();
    auto rhs = op.getRhs().getDefiningOp();
    return (lhs && hasWriteBetween(a, lhs, dom)) ||
           (rhs && hasWriteBetween(a, rhs, dom));
  } else if (auto op = ::mlir::dyn_cast<::imex::dist::SubviewOp>(b)) {
    auto dop = op.getSource().getDefiningOp();
    return dop && hasWriteBetween(a, dop, dom);
  } else {
    std::cerr << "oops. Unexpected op found: ";
    b->dump();
    assert(false);
  }
}

} // namespace dist
} // namespace imex

#endif // _DIST_UTILS_H_INCLUDED_
