//===- PTensorDist.cpp - PTensorToDist Transform  ---------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transform of the PTensor dialect to a combination of
/// PTensor and Dist dialects.
///
/// PTensor operations will stay untouched unless operands are distributed
/// PTensors. PTensors are converted do DistTensorTypes by creation functions,
/// for example by reacting on an input argument 'team'. When creating a
/// DistTensor additional information is attached which provides information to
/// perform distributed operations, such as shape and offsets of the local
/// partition. If operations work on distributed tensors necessary communication
/// with the runtime is performed to identify the local partition. The local
/// tensor is extracted/created and the operation is re-issued for the local
/// part. No deep recursion happens because the operands for the newly created
/// ptensor operations are not distributed. Finally additional ops are added of
/// more communication with the runtime is needed, for example to perform a
/// final global reduction.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Conversion/LLVMCommon/TypeConverter.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Rewrite/FrozenRewritePatternSet.h>

#include "PassDetail.h"

#include <iostream>

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Some helper functions ***
// *******************************

// extract RankedTensor and create ::imex::dist::AllReduceOp
inline ::mlir::Value createAllReduce(::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Attribute op,
                                     ::mlir::Value pTnsr) {
  auto pTnsrTyp = pTnsr.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(pTnsrTyp);
  auto rTnsr = builder.create<::imex::ptensor::ExtractMemRefOp>(
      loc, pTnsrTyp.getMemRefType(), pTnsr);
  return builder.create<::imex::dist::AllReduceOp>(loc, rTnsr.getType(), op,
                                                   rTnsr);
}

// *******************************
// ***** Individual patterns *****
// *******************************

// Base-class for RewriterPatterns which handle recursion
// All our rewriters replace ops with series of ops including the
// op-type which gets rewritten. Rewriters will not rewrite (stop recursion)
// if input PTensor operands are not distributed.
template <typename T>
struct RecOpRewritePattern : public ::mlir::OpRewritePattern<T> {
  using ::mlir::OpRewritePattern<T>::OpRewritePattern;
  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    RecOpRewritePattern<T>::setHasBoundedRewriteRecursion();
  }
};

/// Rewriting ::imex::ptensor::ExtractMemRefOp
/// Get PTensor from DistTensor and apply to ExtractMemRefOp.
struct DistExtractMemRefOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ExtractMemRefOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractMemRefOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // get input
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp) {
      return ::mlir::failure();
    }
    auto pTnsr = createLocalTensorOf(loc, rewriter, op.getInput());
    rewriter.replaceOpWithNewOp<::imex::ptensor::ExtractMemRefOp>(
        op, inpPtTyp.getPTensorType().getMemRefType(), pTnsr);
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::SubviewOp
/// 1. Compute local slice and offsets of dst
/// 2. Apply SubviewOp to subslice and local partition
/// 3. Create new DistTensor
struct DistSubviewOpRWP
    : public RecOpRewritePattern<::imex::ptensor::SubviewOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::SubviewOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // get input and type
    auto src = op.getSource();
    auto inpDTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto outDTTyp = op.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto outPTTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();

    if ((!inpDTTyp && outPTTyp) ||
        outDTTyp) { //} || (op->hasOneUse() &&
                    // op->user_begin()->getName().getStringRef()
                    //== "dist.init_dist_tensor")) {
      return ::mlir::failure();
    }

    auto vDTTYp =
        ::imex::dist::DistTensorType::get(rewriter.getContext(), outPTTyp);
    rewriter.replaceOpWithNewOp<::imex::dist::SubviewOp>(
        op, vDTTYp, src, op.getOffsets(), op.getSizes(), op.getStrides(),
        op.getStaticOffsets(), op.getStaticSizes(), op.getStaticStrides(),
        ::mlir::ValueRange{}, ::mlir::ValueRange{});

    return ::mlir::success();
  }
};

struct DistLoadOpRWP : public RecOpRewritePattern<::imex::ptensor::LoadOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::LoadOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // get input and type
    auto src = op.getArray();
    auto inpDTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();

    if (inpDTTyp) {
      return ::mlir::failure();
    }

    rewriter.replaceOpWithNewOp<::imex::dist::LoadOp>(op, src, op.getIndices());

    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::InsertSliceOp
/// 1. Compute local slice of dst
/// 2. Get local PTensors (dst and src)
/// 3. Apply to ::imex::ptensor::InsertSliceOp
struct DistInsertSliceOpRWP
    : public RecOpRewritePattern<::imex::ptensor::InsertSliceOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::InsertSliceOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // check if inputs are DistTensors
    auto dst = op.getDestination();
    auto src = op.getSource();
    auto dstPTTyp = dst.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto srcPTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dstPTTyp || !srcPTTyp) {
      return ::mlir::failure();
    }

    auto slcOffs = op.getOffsets();
    auto slcSizes = op.getSizes();
    auto slcStrides = op.getStrides();

    // get target slice info
    auto tSlice = rewriter.create<::imex::dist::LocalTargetOfSliceOp>(
        loc, dst, slcOffs, slcSizes, slcStrides);
    ::mlir::ValueRange tSlcOffs = tSlice.getTOffsets();
    auto tSlcSizes = tSlice.getTSizes();

    // Repartition source
    // notice: the source should have the same size as the target
    //         this means there is no additional global offset in the source
    //         so there is no need to incorporate global offsets or alike
    auto rpSrc = createRePartition(loc, rewriter, src, tSlcOffs, tSlcSizes);

    rewriter.replaceOpWithNewOp<::imex::dist::InsertSliceOp>(
        op, dst, rpSrc, op.getOffsets(), op.getSizes(), op.getStrides(),
        tSlcOffs, tSlcSizes);

    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::ARangeOp to get a distributed arange if
/// applicable. Create global, distributed output Tensor as defined by operands.
/// The local partition (e.g. a RankedTensor) are wrapped in a
/// non-distributed PTensor and re-applied to arange op.
/// op gets replaced with global DistTensor
struct DistARangeOpRWP : public RecOpRewritePattern<::imex::ptensor::ARangeOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ARangeOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // nothing to do if no team
    auto team = op.getTeam();
    if (!team)
      return ::mlir::failure();

    // get operands
    auto start = easyIdx(loc, rewriter, op.getStart());
    auto stop = easyIdx(loc, rewriter, op.getStop());
    auto step = easyIdx(loc, rewriter, op.getStep());
    // compute global count (so we know the shape)
    auto count = createCountARange(rewriter, loc, start, stop, step);
    // get number of procs and prank
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);

    // get local shape and offsets
    auto lPart = rewriter.create<::imex::dist::LocalPartitionOp>(
        loc, nProcs, pRank, ::mlir::ValueRange{count});
    auto lShape = lPart.getLShape();
    auto lOffs = lPart.getLOffsets();

    // we can now compute local arange
    auto lSz = easyIdx(loc, rewriter, lShape[0]);
    auto off = easyIdx(loc, rewriter, lOffs[0]);
    start = start + (off * step);
    // create stop
    stop = start + (step * lSz); // start + (lShape[0] * step)
    // finally create local arange
    auto arres = rewriter.create<::imex::ptensor::ARangeOp>(
        loc, start.get(), stop.get(), step.get(), op.getDevice(), nullptr);

    rewriter.replaceOp(
        op, createDistTensor(loc, rewriter, arres, true, {count}, lOffs, team));
    return ::mlir::success();
  }
};

/// Rewriting ::imex::ptensor::CreateOp to get a distributed CreateOp if
/// applicable. Create global, distributed output Tensor as defined by operands.
/// The local partition (e.g. a RankedTensor) are wrapped in a
/// non-distributed PTensor and re-applied to CreateOp.
/// op gets replaced with global DistTensor
struct DistCreateOpRWP : public RecOpRewritePattern<::imex::ptensor::CreateOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::CreateOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // nothing to do if no team
    auto team = op.getTeam();
    if (!team)
      return ::mlir::failure();

    auto gShape = op.getShape();
    // get number of procs and prank
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    // get local shape and offsets
    auto lPart = rewriter.create<::imex::dist::LocalPartitionOp>(loc, nProcs,
                                                                 pRank, gShape);

    // finally create local array
    auto arres = rewriter.create<::imex::ptensor::CreateOp>(
        loc, lPart.getLShape(), op.getDType(), op.getValue(), op.getDevice(),
        nullptr);

    rewriter.replaceOp(op, createDistTensor(loc, rewriter, arres, true, gShape,
                                            lPart.getLOffsets(), team));
    return ::mlir::success();
  }
};

/// Rewrite ::imex::ptensor::EWBinOp to get a distributed ewbinop
/// if operands are distributed.
/// Create global, distributed output tensor with same shape as operands.
/// The local partitions of operands (e.g. RankedTensor) are wrapped in
/// non-distributed PTensors and re-applied to ewbinop.
/// op gets replaced with global DistTensor
struct DistEWBinOpRWP : public RecOpRewritePattern<::imex::ptensor::EWBinOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::mlir::PatternRewriter &rewriter) const override {

    // get input and type
    auto lhsDTTyp =
        op.getLhs().getType().dyn_cast<::imex::dist::DistTensorType>();
    auto rhsDTTyp =
        op.getRhs().getType().dyn_cast<::imex::dist::DistTensorType>();
    auto outDTTyp =
        op.getResult().getType().dyn_cast<::imex::dist::DistTensorType>();
    auto outPTTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();

    if (((!lhsDTTyp || !rhsDTTyp) && outPTTyp) ||
        outDTTyp) { //} || (op->hasOneUse() &&
                    // op->user_begin()->getName().getStringRef() ==
                    //"dist.init_dist_tensor")) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();

    // repartition if necessary

#if 0
    ::mlir::ValueRange tOffs, tSizes;

    if(lhsDTTyp.getPTensorType().getRank() > 0 || rhsDTTyp.getPTensorType().getRank() > 0) {
      auto tPart = createLocalPartition(loc, rewriter, op.getLhs());
      tOffs = tPart.getLOffsets();
      tSizes = tPart.getLShape();
    }
#endif

    auto rbLhs =
        lhsDTTyp.getPTensorType().getRank() == 0
            ? op.getLhs()
            : createRePartition(loc, rewriter, op.getLhs()); //, tOffs, tSizes);
    auto rbRhs =
        rhsDTTyp.getPTensorType().getRank() == 0
            ? op.getRhs()
            : createRePartition(loc, rewriter, op.getRhs()); //, tOffs, tSizes);

    // replace with dist version of ewbinop
    auto vDTTYp =
        ::imex::dist::DistTensorType::get(rewriter.getContext(), outPTTyp);
    rewriter.replaceOpWithNewOp<::imex::ptensor::EWBinOp>(
        op, vDTTYp, op.getOp(), rbLhs, rbRhs);

    return ::mlir::success();
  }
};

/// Rewrite ::imex::ptensor::ReductionOp to get a distributed
/// reduction if operand is distributed.
/// Create global, distributed 0d output tensor.
/// The local partitions of operand (e.g. RankedTensor) is wrapped in
/// non-distributed PTensor and re-applied to reduction.
/// The result is then applied to a distributed allreduce.
/// op gets replaced with global DistTensor
struct DistReductionOpRWP
    : public RecOpRewritePattern<::imex::ptensor::ReductionOp> {
  using RecOpRewritePattern::RecOpRewritePattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    // FIXME reduction over individual dimensions is not supported
    auto loc = op.getLoc();
    // get input
    auto inpDtTyp =
        op.getInput().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpDtTyp) {
      return ::mlir::failure();
    }

    // Local reduction
    auto local = createLocalTensorOf(loc, rewriter, op.getInput());
    // return type 0d with same dtype as input
    auto dtype = inpDtTyp.getPTensorType().getElementType();
    auto retPtTyp = ::imex::ptensor::PTensorType::get({}, dtype);
    auto redPTnsr = rewriter.create<::imex::ptensor::ReductionOp>(
        loc, retPtTyp, op.getOp(), local);
    // global reduction
    auto retRTnsr = createAllReduce(loc, rewriter, op.getOp(), redPTnsr);
    // get global shape, offsets and team
    // result shape is 0d
    auto team = createTeamOf(loc, rewriter, op.getInput());
    // and init our new dist tensor
    auto dmy = ::imex::createInt<1>(loc, rewriter, 0); // FIXME
    auto resPTnsr = rewriter.create<::imex::ptensor::MkPTensorOp>(
        loc, false, retRTnsr, dmy);
    rewriter.replaceOp(
        op, createDistTensor(loc, rewriter, resPTnsr, true, {}, {}, team));
    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct PTensorDistPass : public ::imex::PTensorDistBase<PTensorDistPass> {

  PTensorDistPass() = default;

  void runOnOperation() override {

    ::mlir::FrozenRewritePatternSet patterns;
    insertPatterns<DistARangeOpRWP, DistCreateOpRWP, DistEWBinOpRWP,
                   DistReductionOpRWP, DistExtractMemRefOpRWP, DistSubviewOpRWP,
                   DistInsertSliceOpRWP>(getContext(), patterns);
    (void)::mlir::applyPatternsAndFoldGreedily(this->getOperation(), patterns);

    // groupOps<::imex::dist::SubviewOp>(
    //     this->getAnalysis<::mlir::DominanceInfo>(), this->getOperation(),
    //     [](::imex::dist::SubviewOp &op) { return true; },
    //     [](::imex::dist::SubviewOp &op) { return op.getOperands(); },
    //     [](::imex::dist::SubviewOp &, ::imex::dist::SubviewOp &) {
    //       return false;
    //     }, hasWriteBetween);
  }
};

} // namespace
} // namespace dist

/// Populate the given list with patterns that eliminate Dist ops
void populatePTensorDistPatterns(::mlir::LLVMTypeConverter &converter,
                                 ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createPTensorDistPass() {
  return std::make_unique<::imex::dist::PTensorDistPass>();
}

} // namespace imex
