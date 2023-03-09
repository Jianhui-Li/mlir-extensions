//===- DistToStandard.cpp - DistToStandard conversion  ----------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistToStandard conversion, converting the Dist
/// dialect to standard dialects (including PTensor).
/// Some operations get converted to runtime calls, others with standard
/// MLIR operations from dialects like arith and tensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/DistToStandard/DistToStandard.h>
#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Utils/Utils.h>
#include <imex/Utils/ArithUtils.h>
#include <imex/Utils/PassUtils.h>
#include <imex/Utils/PassWrapper.h>

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/BuiltinOps.h>

#include <array>
#include <iostream>

#include "../PassDetail.h"

using ::imex::ptensor::createDType;

namespace imex {
namespace dist {
namespace {

// create function prototype fo given function name, arg-types and
// return-types
inline void requireFunc(::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        ::mlir::ModuleOp module, const char *fname,
                        ::mlir::TypeRange args, ::mlir::TypeRange results) {
  mlir::OpBuilder::InsertionGuard guard(builder);
  // Insert before module terminator.
  builder.setInsertionPoint(module.getBody(),
                            std::prev(module.getBody()->end()));
  auto funcType = builder.getFunctionType(args, results);
  auto func = builder.create<::mlir::func::FuncOp>(loc, fname, funcType);
  func.setPrivate();
}

// *******************************
// ***** Individual patterns *****
// *******************************

/// Convert a global dist::SubviewOP to ptensor::SubViewOp on the local data.
/// Potentially computes local part if no target part is provided.
/// Even though the op accepts static offs/sizes all computation
/// is done on values - only static dim-sizes of 1 are properly propagated.
/// Static strides are always propagated to PTensor.
struct SubviewOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::SubviewOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::SubviewOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::SubviewOp op,
                  ::imex::dist::SubviewOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get input and type
    auto src = op.getSource();
    auto inpDTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto resDTTyp =
        op.getResult().getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpDTTyp || !resDTTyp) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    // Get the local part of the global slice, team, rank, offsets
    auto _slcOffs = adaptor.getOffsets();
    auto _slcSizes = adaptor.getSizes();
    auto _slcStrides = adaptor.getStrides();
    auto sSlcOffs = adaptor.getStaticOffsets();
    auto sSlcSizes = adaptor.getStaticSizes();
    auto sSlcStrides = adaptor.getStaticStrides();
    ::mlir::ValueRange tOffs = adaptor.getTargetOffsets();
    ::mlir::ValueRange tSizes = adaptor.getTargetSizes();
    auto rank = std::max(sSlcOffs.size(), _slcOffs.size());

    // get offs, sizes strides as values
    auto slcOffs = getMixedAsValues(loc, rewriter, _slcOffs, sSlcOffs);
    auto slcSizes = getMixedAsValues(loc, rewriter, _slcSizes, sSlcSizes);
    auto slcStrides = getMixedAsValues(loc, rewriter, _slcStrides, sSlcStrides);

    if (tOffs.empty()) {
      assert(tSizes.empty());
      auto lTarget = rewriter.create<::imex::dist::LocalTargetOfSliceOp>(
          loc, src, slcOffs, slcSizes, slcStrides);
      tOffs = lTarget.getTOffsets();
      tSizes = lTarget.getTSizes();
    }

    // Compute local part of slice
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);
    auto lSlice = rewriter.create<::imex::dist::LocalOffsetForTargetSliceOp>(
        loc, lOffs, tOffs, slcOffs, slcStrides);
    auto lSlcOffsets = lSlice.getLOffsets();

    // get static size==1 and strides back
    ::mlir::SmallVector<::mlir::OpFoldResult> vOffs, vSizes, vStrides;
    for (size_t i = 0; i < rank; ++i) {
      vOffs.emplace_back(lSlcOffsets[i]);
      vSizes.emplace_back(sSlcSizes[i] == 1
                              ? ::mlir::OpFoldResult{rewriter.getIndexAttr(1)}
                              : ::mlir::OpFoldResult{tSizes[i]});
      auto s = sSlcStrides[i];
      vStrides.emplace_back(
          ::mlir::ShapedType::isDynamic(s)
              ? ::mlir::OpFoldResult{slcStrides[i]}
              : ::mlir::OpFoldResult{rewriter.getIndexAttr(s)});
    }

    // create local view
    auto lTnsr = createLocalTensorOf(loc, rewriter, src);
    auto lView = rewriter.create<::imex::ptensor::SubviewOp>(
        loc, resDTTyp.getPTensorType(), lTnsr, vOffs, vSizes, vStrides);

    // init our new dist tensor
    auto team = createTeamOf(loc, rewriter, src);
    rewriter.replaceOp(op, createDistTensor(loc, rewriter, lView, false,
                                            slcSizes, tOffs, team));
    return ::mlir::success();
  }
};

// Adjusted from NTensor
struct LoadOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LoadOp> {
  using ::mlir::OpConversionPattern<::imex::dist::LoadOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LoadOp op,
                  ::imex::dist::LoadOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get input and type
    auto src = op.getArray();
    auto inpDTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpDTTyp) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    // Get the local part of the global slice, team, rank, offsets
    auto slcOffs = op.getIndices();
    auto rank = slcOffs.size();
    auto one = createIndex(loc, rewriter, 1);
    ::mlir::SmallVector<::mlir::Value> slcSizes(rank, one),
        slcStrides(rank, one);
    ::mlir::ValueRange tOffs = op.getTargetOffsets();

    if (tOffs.empty()) {
      auto lTarget = rewriter.create<::imex::dist::LocalTargetOfSliceOp>(
          loc, src, slcOffs, slcSizes, slcStrides);
      tOffs = lTarget.getTOffsets();
    }

    // Compute local part of slice
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);
    auto lSlice = rewriter.create<::imex::dist::LocalOffsetForTargetSliceOp>(
        loc, lOffs, tOffs, slcOffs, slcStrides);
    auto lSlcOffsets = lSlice.getLOffsets();

    // create local view
    auto lTnsr = createLocalTensorOf(loc, rewriter, src);
    rewriter.replaceOpWithNewOp<::imex::ptensor::LoadOp>(
        op, op.getResult().getType(), lTnsr, lSlcOffsets);

    return ::mlir::success();
  }
};

/// Convert a global dist::InsertSliceOP to ptensor::InsertSliceOp on the local
/// data. Assumes that the input is properly partitioned: the target part or of
/// none provided the default partitioning.
struct InsertSliceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::InsertSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::InsertSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::InsertSliceOp op,
                  ::imex::dist::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto dst = op.getDestination();
    auto src = op.getSource();
    auto dstPTTyp = dst.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto srcPTTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dstPTTyp || !srcPTTyp) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto slcOffs = op.getOffsets();
    auto slcSizes = op.getSizes();
    auto slcStrides = op.getStrides();
    ::mlir::ValueRange tOffs = op.getTargetOffsets();
    ::mlir::ValueRange tSizes = op.getTargetSizes();

    if (tOffs.empty()) {
      // get slice info and create distributed view
      auto lSlice = rewriter.create<::imex::dist::LocalTargetOfSliceOp>(
          loc, dst, slcOffs, slcSizes, slcStrides);
      tOffs = lSlice.getTOffsets();
      tSizes = lSlice.getTSizes();
    }

    // translate target slice offs to relative local
    auto lOffs = createLocalOffsetsOf(loc, rewriter, dst);
    auto rank = tSizes.size();
    ::mlir::SmallVector<::mlir::Value> lSlcOffsets;
    for (unsigned i = 0; i < rank; ++i) {
      auto tOff = easyIdx(loc, rewriter, tOffs[i]);
      auto stride = easyIdx(loc, rewriter, slcStrides[i]);
      auto sOff = easyIdx(loc, rewriter, slcOffs[i]);
      auto lOff = easyIdx(loc, rewriter, lOffs[i]);
      auto zero = easyIdx(loc, rewriter, 0);
      lSlcOffsets.emplace_back((sOff + (tOff * stride) - lOff).max(zero).get());
    }

    // get local ptensors
    auto lDst = createLocalTensorOf(loc, rewriter, dst);
    auto lSrc = createLocalTensorOf(loc, rewriter, src);

    // auto zero = createIndex(loc, rewriter, 0);
    // rewriter.create<::mlir::func::CallOp>(
    //     loc, "_idtr_extractslice",
    //             ::mlir::TypeRange{},
    //             ::mlir::ValueRange{createExtractPtrFromMemRefFromValues(rewriter,
    //             loc, lSlcOffsets),
    //                                createExtractPtrFromMemRefFromValues(rewriter,
    //                                loc, tSizes),
    //                                createExtractPtrFromMemRefFromValues(rewriter,
    //                                loc, slcStrides), zero, zero, zero, zero,
    //                                zero}
    //             );

    // apply to InsertSliceOp
    rewriter.replaceOpWithNewOp<::imex::ptensor::InsertSliceOp>(
        op, lDst, lSrc, lSlcOffsets, tSizes, slcStrides);

    return ::mlir::success();
  }
};

/// Convert a global dist::EWBinOp to ptensor::EWBinOp on the local data.
/// Assumes that the partitioning of the inputs are properly aligned.
struct EWBinOpConverter
    : public ::mlir::OpConversionPattern<::imex::ptensor::EWBinOp> {
  using ::mlir::OpConversionPattern<
      ::imex::ptensor::EWBinOp>::OpConversionPattern;

  /// Initialize the pattern.
  void initialize() {
    /// Signal that this pattern safely handles recursive application.
    setHasBoundedRewriteRecursion();
  }

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::imex::ptensor::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto lhs = op.getLhs();
    auto rhs = op.getRhs();
    auto lhsDtTyp = lhs.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto rhsDtTyp = rhs.getType().dyn_cast<::imex::dist::DistTensorType>();
    auto resDtTyp =
        op.getResult().getType().dyn_cast<::imex::dist::DistTensorType>();
    // return failure if wrong ops or not distributed
    if (!lhsDtTyp || !rhsDtTyp || !resDtTyp) {
      return ::mlir::failure();
    }

    auto lhsPtTyp = lhsDtTyp.getPTensorType();
    // auto rhsPtTyp = rhsDtTyp.getPTensorType();
    auto resPtTyp = resDtTyp.getPTensorType();

    auto loc = op.getLoc();
    // local ewb operands
    auto lLhs = createLocalTensorOf(loc, rewriter, lhs);
    auto lRhs = createLocalTensorOf(loc, rewriter, rhs);

    // return type same as lhs for now
    auto ewbres = rewriter.create<::imex::ptensor::EWBinOp>(
        loc, resPtTyp, op.getOp(), lLhs, lRhs);

    // get global shape, offsets and team
    auto ref = resPtTyp.getRank() == lhsPtTyp.getRank() ? lhs : rhs;
    auto team = createTeamOf(loc, rewriter, ref);
    auto gShape = createGlobalShapeOf(loc, rewriter, ref);
    auto lPart = createLocalPartition(loc, rewriter, ref, team, gShape);
    // and init our new dist tensor
    rewriter.replaceOp(op, createDistTensor(loc, rewriter, ewbres, true, gShape,
                                            lPart.getLOffsets(), team));

    return ::mlir::success();
  }
};

// RuntimePrototypesOp -> func.func ops
// adding required function prototypes to the module level
struct RuntimePrototypesOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::RuntimePrototypesOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::RuntimePrototypesOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RuntimePrototypesOp op,
                  ::imex::dist::RuntimePrototypesOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto mod = op->getParentOp();
    assert(::mlir::isa<mlir::ModuleOp>(mod));
    ::mlir::ModuleOp module = ::mlir::cast<mlir::ModuleOp>(mod);
    // auto dtype = rewriter.getI64Type();
    auto indexType = rewriter.getIndexType();
    auto dtypeType = rewriter.getIntegerType(sizeof(int) * 8);
    auto rankType = indexType;
    auto opType =
        rewriter.getIntegerType(sizeof(::imex::ptensor::ReduceOpId) * 8);
    auto mrType = ::mlir::UnrankedMemRefType::get(rewriter.getI64Type(), {});
    requireFunc(loc, rewriter, module, "printMemrefI64", {mrType}, {});
    auto IdxmrType =
        ::mlir::UnrankedMemRefType::get(rewriter.getIndexType(), {});
    requireFunc(loc, rewriter, module, "printMemrefInd", {IdxmrType}, {});
    requireFunc(loc, rewriter, module, "_idtr_nprocs", {indexType},
                {indexType});
    requireFunc(loc, rewriter, module, "_idtr_prank", {indexType}, {indexType});
    requireFunc(loc, rewriter, module, "_idtr_reduce_all",
                {indexType, indexType, indexType, indexType, dtypeType, opType},
                {});
    requireFunc(loc, rewriter, module, "_idtr_repartition",
                // rank, gShapePtr, dtype, lDataPtr, lOffsPtr, lShapePtr,
                // lStridesPtr, offsPtr, szsPtr, outPtr, team
                {rankType, indexType, dtypeType, indexType, indexType,
                 indexType, indexType, indexType, indexType, indexType,
                 indexType},
                {});
    requireFunc(loc, rewriter, module, "_idtr_extractslice",
                {indexType, indexType, indexType, indexType, indexType,
                 indexType, indexType, indexType},
                {});

    rewriter.eraseOp(op);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::NProcsOp into runtime call to _idtr_nprocs
struct NProcsOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::NProcsOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::NProcsOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::NProcsOp op,
                  ::imex::dist::NProcsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, "_idtr_nprocs", rewriter.getIndexType(), adaptor.getTeam());
    return ::mlir::success();
  }
};

// Convert ::imex::dist::PRankOp into runtime call to _idtr_prank
struct PRankOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::PRankOp> {
  using ::mlir::OpConversionPattern<::imex::dist::PRankOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::PRankOp op,
                  ::imex::dist::PRankOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<::mlir::func::CallOp>(
        op, "_idtr_prank", rewriter.getIndexType(), adaptor.getTeam());
    return ::mlir::success();
  }
};

/// Replace ::imex::dist::InitDistTensorOp with unrealized_conversion_cast
/// InitDistTensorOp is a dummy op used only for propagating dist infos
struct InitDistTensorOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::InitDistTensorOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::InitDistTensorOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::InitDistTensorOp op,
                  ::imex::dist::InitDistTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto dtType = op.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dtType)
      return ::mlir::failure();

    auto rank = dtType.getPTensorType().getRank();
    auto lOffs = adaptor.getLOffsets();
    ::mlir::SmallVector<::mlir::Value> oprnds(3 + rank + lOffs.size());

    // we use enum values to make sure we use the correct order in the cast
    oprnds[LTENSOR] = adaptor.getPTensor();
    oprnds[TEAM] = adaptor.getTeam();
    oprnds[BALANCED] =
        createIndex(op.getLoc(), rewriter, adaptor.getBalancedAttr().getInt());
    auto gShape = adaptor.getGShape();
    assert(GSHAPE + 1 == LOFFSETS);
    for (auto i = 0; i < rank; ++i) {
      oprnds[GSHAPE + i] = gShape[i];
    }
    if (lOffs.size()) {
      assert(lOffs.size() == (size_t)rank);
      for (auto i = 0; i < rank; ++i) {
        oprnds[GSHAPE + rank + i] = lOffs[i];
      }
    }
    rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, dtType, ::mlir::ValueRange{oprnds});
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::ExtractFromDistOp into respective operand of defining
/// op. We assume the defining op is either InitDistTensorOp or it is an
/// block-argument which was converted by a unrealized_conversion_cast.
// we use enum values to make sure we use the correct order in the cast
template <typename OP>
struct ExtractFromDistOpConverter : public ::mlir::OpConversionPattern<OP> {
  using ::mlir::OpConversionPattern<OP>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(OP op, typename OP::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto dt = adaptor.getDTensor();
    auto castOp =
        dt.template getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    while (castOp && castOp->getNumOperands() == 1) {
      dt = castOp->getOperand(0);
      auto nOp =
          dt.template getDefiningOp<::mlir::UnrealizedConversionCastOp>();
      if (nOp) {
        castOp = nOp;
      } else
        break;
    }

    // std::cerr << "defOp: "; op.getDTensor().dump(); std::cerr << std::endl;
    if (castOp && castOp->getNumOperands() > 2) {
      // block args get type-converted into UnrealizedConversionCastOp
      // here this is from a block arg; we can extract operands from the
      // inserted cast
      if constexpr (std::is_same_v<OP, ::imex::dist::LocalTensorOfOp>) {
        rewriter.replaceOp(op, castOp.getInputs()[LTENSOR]);
      } else if constexpr (std::is_same_v<OP, ::imex::dist::TeamOfOp>) {
        rewriter.replaceOp(op, castOp.getInputs()[TEAM]);
      } else if constexpr (std::is_same_v<OP, ::imex::dist::IsBalancedOp>) {
        rewriter.replaceOp(op, castOp.getInputs()[BALANCED]);
      } else if (castOp.getInputs().size() > GSHAPE) {
        // if rank==0 there might be no gshape/loffs args in cast-op
        auto nxt = castOp.getInputs()[GSHAPE];
        auto nxtType = nxt.getType().template dyn_cast<::mlir::MemRefType>();
        auto dTTyp = adaptor.getDTensor()
                         .getType()
                         .template dyn_cast<::imex::dist::DistTensorType>();
        assert(dTTyp);
        auto rank = dTTyp.getPTensorType().getRank();

        if constexpr (std::is_same_v<OP, ::imex::dist::GlobalShapeOfOp>) {
          // Check if this is from block args, e.g. a memref
          if (nxtType) {
            assert(castOp.getInputs().size() == DIST_META_LAST);
            rewriter.replaceOp(
                op, createValuesFromMemRef(rewriter, op.getLoc(),
                                           castOp.getInputs()[GSHAPE]));
          } else {
            // if not a memref, then it originates from a InitTensor
            ::mlir::SmallVector<::mlir::Value> vals(rank);
            for (int64_t i = 0; i < rank; ++i) {
              vals[i] = castOp.getInputs()[GSHAPE + i];
            }
            rewriter.replaceOp(op, vals);
          }
        } else if constexpr (std::is_same_v<OP,
                                            ::imex::dist::LocalOffsetsOfOp>) {
          if (nxtType) {
            assert(castOp.getInputs().size() == DIST_META_LAST);
            rewriter.replaceOp(
                op, createValuesFromMemRef(rewriter, op.getLoc(),
                                           castOp.getInputs()[LOFFSETS]));
          } else {
            // if not a memref, then it originates from a InitTensor
            // offsets are optional
            if (castOp.getInputs().size() > (size_t)(GSHAPE + rank)) {
              ::mlir::SmallVector<::mlir::Value, 4> vals(rank);
              for (int64_t i = 0; i < rank; ++i) {
                vals[i] = castOp.getInputs()[GSHAPE + rank + i];
              }
              rewriter.replaceOp(op, vals);
            } else {
              rewriter.replaceOp(op, ::mlir::ValueRange{});
            }
          }
        } else {
          assert(!"Unknown dist meta item requested");
        }
      } else {
        rewriter.replaceOp(op, ::mlir::ValueRange{});
      }
    } else if (auto defOp = dt.template getDefiningOp<
                            ::imex::dist::InitDistTensorOp>()) {
      // here this is from a normal InitDistTensorOp; we can extract operands
      // from it
      if constexpr (std::is_same_v<OP, ::imex::dist::LocalTensorOfOp>) {
        rewriter.replaceOp(op, defOp.getPTensor());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::TeamOfOp>) {
        rewriter.replaceOp(op, defOp.getTeam());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::IsBalancedOp>) {
        rewriter.replaceOp(op, createIndex(op.getLoc(), rewriter,
                                           defOp.getBalancedAttr().getInt()));
      } else if constexpr (std::is_same_v<OP, ::imex::dist::GlobalShapeOfOp>) {
        rewriter.replaceOp(op, defOp.getGShape());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::LocalOffsetsOfOp>) {
        rewriter.replaceOp(op, defOp.getLOffsets());
      }
    } else {
      // neither an InitDistTensorOp nor a valid cast
      return ::mlir::failure();
    }
    return ::mlir::success();
  }
};

// explicit instantiation with custom name for each meta data
using GlobalShapeOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::GlobalShapeOfOp>;
using LocalTensorOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::LocalTensorOfOp>;
using LocalOffsetsOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::LocalOffsetsOfOp>;
using TeamOfOpConverter = ExtractFromDistOpConverter<::imex::dist::TeamOfOp>;
using IsBalancedOpConverter =
    ExtractFromDistOpConverter<::imex::dist::IsBalancedOp>;

/// Lowering ::imex::dist::LocalPartitionOp: Compute default partition
/// for a given shape and number of processes.
/// We currently assume evenly split data.
struct LocalPartitionOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalPartitionOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalPartitionOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalPartitionOp op,
                  ::imex::dist::LocalPartitionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // FIXME: non-even partitions, ndims
    auto gShape = adaptor.getGShape();
    int64_t rank = (int64_t)gShape.size();

    if (rank == 0) {
      rewriter.eraseOp(op);
      return ::mlir::success();
    }

    auto loc = op.getLoc();
    auto sz = easyIdx(loc, rewriter, gShape.front());
    auto np = easyIdx(loc, rewriter, adaptor.getNumProcs());
    auto pr = easyIdx(loc, rewriter, adaptor.getPRank());
    auto one = easyIdx(loc, rewriter, 1);
    auto zero = easyIdx(loc, rewriter, 0);

    // compute tile size and local size (which can be smaller)
    auto tSz = (sz + np - one) / np;
    auto lOff = sz.min(tSz * pr);
    auto lSz = sz.min(lOff + tSz) - lOff;

    // store in result range
    ::mlir::SmallVector<::mlir::Value> res(2 * rank, zero.get());
    res[0] = lOff.get();
    res[rank] = lSz.get();
    for (int64_t i = 1; i < rank; ++i) {
      res[rank + i] = gShape[i];
    }

    rewriter.replaceOp(op, res);
    return ::mlir::success();
  }
};

// Compute offset from local data to overlap of given slice
// assuming the given target part
struct LocalOffsetForTargetSliceOpConverter
    : public ::mlir::OpConversionPattern<
          ::imex::dist::LocalOffsetForTargetSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalOffsetForTargetSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOffsetForTargetSliceOp op,
                  ::imex::dist::LocalOffsetForTargetSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();

    auto bOffs = adaptor.getBaseOffsets();   // global offset of local base
    auto tOffs = adaptor.getTargetOffsets(); // requested off of target slice
    auto slcOffs = adaptor.getOffsets();
    auto slcStrides = adaptor.getStrides();
    int64_t rank = bOffs.size();

    ::mlir::SmallVector<::mlir::Value> results;

    for (auto i = 0; i < rank; ++i) {
      auto bOff = easyIdx(loc, rewriter, bOffs[i]);
      auto slcOff = easyIdx(loc, rewriter, slcOffs[i]);
      auto slcStride = easyIdx(loc, rewriter, slcStrides[i]);
      auto tOff = easyIdx(loc, rewriter, tOffs[i]);

      auto lOff = slcOff + (tOff * slcStride) - bOff;
      results.emplace_back(lOff.get());
    }

    rewriter.replaceOp(op, results);
    return ::mlir::success();
  }
};

// Compute the overlap of local data and global slice and return
// as target part (global offset/size relative to requested slice)
// Currently only dim0 is cut, hence offs/sizes of all other dims
// will be identical to the ones of the requested slice
// (e.g. same size and offset 0)
struct LocalTargetOfSliceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalTargetOfSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalTargetOfSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalTargetOfSliceOp op,
                  ::imex::dist::LocalTargetOfSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = adaptor.getDTensor(); // global offset of local base
    auto dtType = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dtType)
      return ::mlir::failure();

    auto loc = op.getLoc();
    auto slcOffs = adaptor.getOffsets();
    auto slcSizes = adaptor.getSizes();
    auto slcStrides = adaptor.getStrides();
    int64_t rank = slcOffs.size();

    // Get the local part of the global slice, team, rank, offsets
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);
    auto lTnsr = createLocalTensorOf(loc, rewriter, src);
    auto lMemRef = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, dtType.getPTensorType().getMemRefType(), lTnsr);
    auto lExtnt = ::mlir::linalg::createOrFoldDimOp(rewriter, loc, lMemRef, 0);

    // Get the vals from dim 0
    auto lOff = easyIdx(loc, rewriter, lOffs[0]);
    auto slcOff = easyIdx(loc, rewriter, slcOffs[0]);
    auto slcStride = easyIdx(loc, rewriter, slcStrides[0]);
    auto slcSize = easyIdx(loc, rewriter, slcSizes[0]);

#if HAVE_KDYNAMIC_SIZED_OPS
    if (auto cval = ::mlir::getConstantIntValue(slcSize.get());
        cval && cval == ::mlir::ShapedType::kDynamic) {
      slcSize = easyIdx(loc, rewriter, lExtnt);
    }
#endif

    auto zeroIdx = easyIdx(loc, rewriter, 0);
    auto oneIdx = easyIdx(loc, rewriter, 1);
    EasyVal<bool> eTrue(loc, rewriter, true);
    EasyVal<bool> eFalse(loc, rewriter, false);

    // last index of slice
    auto slcEnd = slcOff + slcSize * slcStride;
    // last index of local partition
    auto lEnd = lOff + easyIdx(loc, rewriter, lExtnt);

    // check if requested slice fully before local partition
    auto beforeLocal = slcEnd.ult(lOff);
    // check if requested slice fully behind local partition
    auto behindLocal = lEnd.ule(slcOff);
    // check if there is overlap
    auto overlaps =
        beforeLocal.select(eFalse, behindLocal.select(eFalse, eTrue));

    auto idxTyp = rewriter.getIndexType();
    auto offSz = rewriter.create<::mlir::scf::IfOp>(
        loc, ::mlir::TypeRange{idxTyp, idxTyp}, overlaps.get(),
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          // start index within our local part if slice starts before lpart
          auto start = (lOff - slcOff + slcStride - oneIdx) / slcStride;
          // start is 0 if starts within local part
          start = slcOff.ult(lOff).select(start, zeroIdx);
          // now compute size within our local part
          auto sz =
              (lEnd.min(slcEnd) - (slcOff + (start * slcStride))) / slcStride;
          builder.create<::mlir::scf::YieldOp>(
              loc, ::mlir::ValueRange{start.get(), sz.get()});
        },
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          auto start = beforeLocal.select(slcSize, zeroIdx);
          builder.create<::mlir::scf::YieldOp>(
              loc, ::mlir::ValueRange{start.get(), zeroIdx.get()});
        });

    ::mlir::SmallVector<::mlir::Value> results(2 * rank, zeroIdx.get());
    results[0 * rank] = offSz.getResult(0);
    results[1 * rank] = offSz.getResult(1);

    for (auto i = 1; i < rank; ++i) {
      // results[0 * rank + i] = slcOffs[i];
#if HAVE_KDYNAMIC_SIZED_OPS
      if (auto cval = ::mlir::getConstantIntValue(slcSizes[i]);
          cval && cval == ::mlir::ShapedType::kDynamic) {
        if (auto oval = ::mlir::getConstantIntValue(slcOffs[i]);
            oval && oval == 0) {
          if (auto sval = ::mlir::getConstantIntValue(slcStrides[i]);
              sval && sval == 1) {
            results[1 * rank + i] =
                ::mlir::linalg::createOrFoldDimOp(rewriter, loc, lMemRef, i);
            continue;
          }
        }
        assert(!"Unspecified end in slice implemented only if slice in dim>0 "
                "is equivalent to '0::1'");
      } else
#endif
        results[1 * rank + i] = slcSizes[i];
    }

    rewriter.replaceOp(op, results);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::AllReduceOp into runtime call to "_idtr_reduce_all".
/// Pass local RankedTensor as argument.
/// Replaces op with new DistTensor.
struct AllReduceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::AllReduceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::AllReduceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::AllReduceOp op,
                  ::imex::dist::AllReduceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto mRef = adaptor.getData();
    auto mRefType = mRef.getType().dyn_cast<::mlir::MemRefType>();
    if (!mRefType)
      return ::mlir::failure();

    auto rank = createIndex(loc, rewriter, mRefType.getRank());
    auto opV = rewriter.create<::mlir::arith::ConstantOp>(loc, op.getOp());
    auto dtype = createDType(loc, rewriter, mRefType);

    auto fsa = rewriter.getStringAttr("_idtr_reduce_all");
    auto meta =
        rewriter.create<::mlir::memref::ExtractStridedMetadataOp>(loc, mRef);
    auto dataPtr = createExtractPtrFromMemRef(rewriter, loc, mRef, meta);

    auto sizePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, meta.getSizes());
    auto stridePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, meta.getStrides());

    rewriter.create<::mlir::func::CallOp>(
        loc, fsa, ::mlir::TypeRange(),
        ::mlir::ValueRange({rank, dataPtr, sizePtr, stridePtr, dtype, opV}));
    rewriter.replaceOp(op, mRef);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::LocalBoundingBoxOp
/// Takes incoming (optional) bounding box and extends if provided target part
/// starts earlier and/or ends later.
/// bounding box and target parts are represented as offs and sizes.
struct LocalBoundingBoxOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalBoundingBoxOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalBoundingBoxOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalBoundingBoxOp op,
                  ::imex::dist::LocalBoundingBoxOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    /*
      - compute global start-offset and end of each view and dimension
      - compute bounding box containing all views: [min(start), max(end)[ for
      each dim
    */
    auto base = op.getBase();

    auto dTTyp = base.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dTTyp)
      return ::mlir::failure();

    auto loc = op.getLoc();
    auto vOffs = op.getOffsets();
    // auto vSizes = op.getSizes();
    auto vStrides = op.getStrides();
    auto tOffs = op.getTargetOffsets();
    auto tSizes = op.getTargetSizes();
    auto bbOffs = op.getBBOffsets();
    auto bbSizes = op.getBBSizes();
    bool hasBB = !bbOffs.empty();

    auto rank = vOffs.size();

    // min start index (among all views) for each dim followed by sizes
    ::mlir::SmallVector<::mlir::Value> oprnds(rank * 2);

    // for each dim and view compute min offset and max end
    for (size_t i = 0; i < rank; ++i) {
      ::mlir::SmallVector<EasyIdx> doffs;
      ::mlir::SmallVector<EasyIdx> dends;
      auto tOff = easyIdx(loc, rewriter, tOffs[i]);
      auto tSz = easyIdx(loc, rewriter, tSizes[i]);
      auto vOff = easyIdx(loc, rewriter, vOffs[i]);
      auto st = easyIdx(loc, rewriter, vStrides[i]);
      auto off = vOff + tOff;
      auto bbOff = hasBB ? easyIdx(loc, rewriter, bbOffs[i]) : off;
      auto end = off + (tSz * st);
      auto bbEnd = hasBB ? bbOff + easyIdx(loc, rewriter, bbSizes[i]) : end;
      bbOff = bbOff.min(off);
      auto bbSz = bbEnd.max(end) - bbOff;
      oprnds[i] = bbOff.get();
      oprnds[i + rank] = bbSz.get();
    }

    rewriter.replaceOp(op, oprnds);
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::RePartitionOp
/// Creates a new tensor from the input tensor by re-partitioning it
/// according to the target part (or default). The repartitioning
/// itself happens in a library call.
struct RePartitionOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::RePartitionOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::RePartitionOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::RePartitionOp op,
                  ::imex::dist::RePartitionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto base = op.getBase();

    auto dTTyp = base.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!dTTyp)
      return ::mlir::failure();

    auto loc = op.getLoc();
    auto rank = dTTyp.getPTensorType().getRank();
    auto elTyp = dTTyp.getPTensorType().getElementType();
    ::mlir::ValueRange tOffs = op.getTargetOffsets();
    ::mlir::ValueRange tSizes = op.getTargetSizes();

    // Get required info from base
    auto team = createTeamOf(loc, rewriter, base);
    auto gShape = createGlobalShapeOf(loc, rewriter, base);
    auto lOffs = createLocalOffsetsOf(loc, rewriter, base);
    auto lTnsr = createLocalTensorOf(loc, rewriter, base);

    // default target partition is balanced
    if (tSizes.empty()) {
      auto lPart = createLocalPartition(loc, rewriter, base, team, gShape);
      tOffs = lPart.getLOffsets();
      tSizes = lPart.getLShape();
    }

    // Now it's time to get memrefs their pointers for the function call
    auto bMRTyp = dTTyp.getPTensorType().getMemRefType();
    auto bMRef =
        rewriter.create<::imex::ptensor::ExtractMemRefOp>(loc, bMRTyp, lTnsr);
    auto bMeta =
        rewriter.create<::mlir::memref::ExtractStridedMetadataOp>(loc, bMRef);
    auto lDataPtr = createExtractPtrFromMemRef(rewriter, loc, bMRef, bMeta);
    auto lShapePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, bMeta.getSizes());
    auto lStridesPtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, bMeta.getStrides());
    auto gShapePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, gShape);
    auto lOffsPtr = createExtractPtrFromMemRefFromValues(rewriter, loc, lOffs);

    // make memrefs and get pointers for offset and sizes
    auto offsPtr = createExtractPtrFromMemRefFromValues(rewriter, loc, tOffs);
    auto szsPtr = createExtractPtrFromMemRefFromValues(rewriter, loc, tSizes);

    // create output tensor with target size
    auto outTnsr = rewriter.create<::imex::ptensor::CreateOp>(
        loc, tSizes, ::imex::ptensor::fromMLIR(elTyp), nullptr, nullptr,
        nullptr); // FIXME device
    auto outMRTyp = outTnsr.getType()
                        .dyn_cast<::imex::ptensor::PTensorType>()
                        .getMemRefType();
    auto outMR = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, outMRTyp, outTnsr);
    auto outPtr = createExtractPtrFromMemRef(rewriter, loc, outMR);

    // call our runtime function to redistribute data across processes
    auto rankV = createIndex(loc, rewriter, rank);
    auto dtype = createDType(loc, rewriter, bMRTyp);
    auto fun = rewriter.getStringAttr("_idtr_repartition");
    (void)rewriter.create<::mlir::func::CallOp>(
        loc, fun, ::mlir::TypeRange(),
        ::mlir::ValueRange{rankV, gShapePtr, dtype, lDataPtr, lOffsPtr,
                           lShapePtr, lStridesPtr, offsPtr, szsPtr, outPtr,
                           team});

    // init dist tensor
    rewriter.replaceOp(op, createDistTensor(loc, rewriter, outTnsr, true,
                                            gShape, tOffs, team));

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Full Pass
struct ConvertDistToStandardPass
    : public ::imex::ConvertDistToStandardBase<ConvertDistToStandardPass> {
  ConvertDistToStandardPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    ::mlir::ValueDecomposer decomposer;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // DistTensor gets converted into its individual members
    auto convDTensor = [&ctxt](::imex::dist::DistTensorType type,
                               ::mlir::SmallVectorImpl<::mlir::Type> &types) {
      const auto off = types.size();
      auto rank = type.getPTensorType().getRank();
      types.resize(off + DIST_META_LAST - (rank ? 0 : 2));
      types[LTENSOR + off] = type.getPTensorType();
      types[TEAM + off] = ::mlir::IndexType::get(&ctxt);
      types[BALANCED + off] = ::mlir::IndexType::get(&ctxt);
      if (rank) {
        auto mrTyp = ::mlir::MemRefType::get(::std::array<int64_t, 1>{rank},
                                             ::mlir::IndexType::get(&ctxt));
        types[LOFFSETS + off] = mrTyp;
        types[GSHAPE + off] = mrTyp;
      }
      return ::mlir::success();
    };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convDTensor);

    /// Convert multiple elements (as converted by the above convDTensor) into a
    /// single DistTensor
    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    // auto materializeDTArg =
    //     [](
    //         ::mlir::OpBuilder &builder, ::imex::dist::DistTensorType type,
    //         ::mlir::ValueRange inputs,
    //         ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
    //   assert(inputs.size() == 4);
    //   return materializeDistTensor(builder, loc, inputs[0], inputs[1],
    //   inputs[2], inputs[3]);
    // };

    // typeConverter.addArgumentMaterialization(materializeDTArg);
    typeConverter.addSourceMaterialization(materializeCast);
    // the inverse of the ArgumentMaterialization splits a DistTensor into
    // multiple return args
    decomposer.addDecomposeValueConversion(
        [](::mlir::OpBuilder &builder, ::mlir::Location loc,
           ::imex::dist::DistTensorType resultType, ::mlir::Value value,
           ::mlir::SmallVectorImpl<::mlir::Value> &values) {
          const auto off = values.size();
          auto rank = resultType.getPTensorType().getRank();

          // we use enum values to make sure we use the correct order in the
          // cast
          values.resize(off + DIST_META_LAST - (rank ? 0 : 2));
          values[LTENSOR + off] = createLocalTensorOf(loc, builder, value);
          values[TEAM + off] = createTeamOf(loc, builder, value);
          values[BALANCED + off] = createIsBalanced(loc, builder, value);
          if (rank) {
            values[GSHAPE + off] = createMemRefFromElements(
                builder, loc, builder.getIndexType(),
                createGlobalShapeOf(loc, builder, value));
            values[LOFFSETS + off] = createMemRefFromElements(
                builder, loc, builder.getIndexType(),
                createLocalOffsetsOf(loc, builder, value));
          }
          return ::mlir::success();
        });

    // No dist should remain
    target.addIllegalDialect<::imex::dist::DistDialect>();
    target.addLegalDialect<
        ::mlir::linalg::LinalgDialect, ::mlir::arith::ArithDialect,
        ::imex::ptensor::PTensorDialect, ::mlir::memref::MemRefDialect,
        ::mlir::scf::SCFDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // make sure function boundaries get converted
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp>(
        [&](::mlir::func::ReturnOp op) {
          return typeConverter.isLegal(op.getOperandTypes());
        });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    target.addDynamicallyLegalOp<::imex::ptensor::EWBinOp>(
        [&](::imex::ptensor::EWBinOp op) { return typeConverter.isLegal(op); });

    // All the dist conversion patterns/rewriter
    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<
        InsertSliceOpConverter, SubviewOpConverter, EWBinOpConverter,
        LocalBoundingBoxOpConverter, RePartitionOpConverter,
        RuntimePrototypesOpConverter, NProcsOpConverter, PRankOpConverter,
        InitDistTensorOpConverter, LocalPartitionOpConverter,
        LocalOffsetForTargetSliceOpConverter, LocalTargetOfSliceOpConverter,
        GlobalShapeOfOpConverter, IsBalancedOpConverter,
        LocalTensorOfOpConverter, LocalOffsetsOfOpConverter, TeamOfOpConverter,
        AllReduceOpConverter>(typeConverter, &ctxt);
    // This enables the function boundary handling with the above
    // converters/materializations
    populateDecomposeCallGraphTypesPatterns(&ctxt, typeConverter, decomposer,
                                            patterns);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }

    // singularize calls to nprocs and prank
    auto singularize = [](::mlir::func::CallOp &op1,
                          ::mlir::func::CallOp &op2) {
      return op1.getOperands()[0] == op2.getOperands()[0];
    };
    groupOps<::mlir::func::CallOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::mlir::func::CallOp &op) {
          return op.getCallee() == "_idtr_nprocs";
        },
        [](::mlir::func::CallOp &op) { return op.getOperands(); }, singularize);
    groupOps<::mlir::func::CallOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::mlir::func::CallOp &op) {
          return op.getCallee() == "_idtr_prank";
        },
        [](::mlir::func::CallOp &op) { return op.getOperands(); }, singularize);
  }
};

} // namespace
} // namespace dist

/// Populate the given list with patterns that convert Dist to Standard
void populateDistToStandardConversionPatterns(
    ::mlir::LLVMTypeConverter &converter, ::mlir::RewritePatternSet &patterns) {
  assert(false);
}

/// Create a pass that convert Dist to Standard
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertDistToStandardPass() {
  return std::make_unique<::imex::dist::ConvertDistToStandardPass>();
}

} // namespace imex
