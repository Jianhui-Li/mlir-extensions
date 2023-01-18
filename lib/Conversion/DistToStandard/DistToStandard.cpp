//===- DistToStandard.cpp - DistToStandard conversion  ----------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DistToStandard conversion, converting the Dist
/// dialect to standard dialects.
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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/DecomposeCallGraphTypes.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinOps.h>

#include <array>
#include <iostream>

#include "../PassDetail.h"

using ::imex::ptensor::createDType;

namespace imex {
namespace dist {
namespace {

template <typename T>
inline auto createExtractPtrFromMemRef(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc, ::mlir::Value mr,
                                       T meta) {
  auto off = easyIdx(loc, builder, meta.getOffset());
  auto aptr = easyIdx(
      loc, builder,
      builder.create<::mlir::memref::ExtractAlignedPointerAsIndexOp>(loc, mr));
  return (aptr + (off * easyIdx(loc, builder, sizeof(uint64_t)))).get();
}

inline auto createExtractPtrFromMemRefFromValues(::mlir::OpBuilder &builder,
                                                 ::mlir::Location loc,
                                                 ::mlir::ValueRange elts) {
  auto mr =
      createMemRefFromElements(builder, loc, builder.getIndexType(), elts);
  auto meta = builder.create<::mlir::memref::ExtractStridedMetadataOp>(loc, mr);
  return createExtractPtrFromMemRef(builder, loc, mr, meta);
}

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

// RuntimePrototypesOp -> func.func ops
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
    auto opType =
        rewriter.getIntegerType(sizeof(::imex::ptensor::ReduceOpId) * 8);

    requireFunc(loc, rewriter, module, "_idtr_nprocs", {indexType},
                {indexType});
    requireFunc(loc, rewriter, module, "_idtr_prank", {indexType}, {indexType});
    // autp mrType = ::mlir::UnrankedMemRefType::get(indexType, {});
    requireFunc(
        loc, rewriter, module, "_idtr_reduce_all",
        // {getMemRefType(rewriter.getContext(), 0, dtype), dtypeType, opType},
        // {::mlir::UnrankedMemRefType::get(dtype, {}), dtypeType, opType}, {});
        {indexType, indexType, indexType, indexType, dtypeType, opType}, {});
    requireFunc(loc, rewriter, module, "_idtr_rebalance",
                {indexType, indexType, indexType, indexType, indexType,
                 indexType, dtypeType, indexType, indexType, indexType,
                 indexType},
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

/// Erase ::imex::dist::InitDistTensorOp; it is a dummy op
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
    ::mlir::SmallVector<::mlir::Value> oprnds(2 + (2 * rank));
    oprnds[LTENSOR] = adaptor.getPTensor();
    oprnds[TEAM] = adaptor.getTeam();
    auto gShape = adaptor.getGShape();
    auto lOffs = adaptor.getLOffsets();
    assert(GSHAPE + 1 == LOFFSETS);
    for (auto i = 0; i < rank; ++i) {
      oprnds[GSHAPE + i] = gShape[i];
      oprnds[GSHAPE + rank + i] = lOffs[i];
    }
    rewriter.replaceOpWithNewOp<::mlir::UnrealizedConversionCastOp>(
        op, dtType, ::mlir::ValueRange{oprnds});
    return ::mlir::success();
  }
};

/// Convert ::imex::dist::ExtractFromDistOp into respective operand of defining
/// op. We assume the defining op is either InitDistTensorOp or it is an
/// block-argument which was converted by a unrealized_conversion_cast.
template <typename OP>
struct ExtractFromDistOpConverter : public ::mlir::OpConversionPattern<OP> {
  using ::mlir::OpConversionPattern<OP>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(OP op, typename OP::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto defOp = op.getDTensor()
                     .template getDefiningOp<::imex::dist::InitDistTensorOp>();
    if (defOp) {
      // here this is from a normal InitDistTensorOp; we can extract operands
      // from it
      if constexpr (std::is_same_v<OP, ::imex::dist::LocalTensorOfOp>) {
        rewriter.replaceOp(op, defOp.getPTensor());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::TeamOfOp>) {
        rewriter.replaceOp(op, defOp.getTeam());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::GlobalShapeOfOp>) {
        rewriter.replaceOp(op, defOp.getGShape());
      } else if constexpr (std::is_same_v<OP, ::imex::dist::LocalOffsetsOfOp>) {
        rewriter.replaceOp(op, defOp.getLOffsets());
      }
    } else {
      // disttensor block args get type-converted into
      // UnrealizedConversionCastOp
      auto castOp =
          adaptor.getDTensor()
              .template getDefiningOp<::mlir::UnrealizedConversionCastOp>();
      if (!castOp)
        return ::mlir::failure();
      // here this is from a block arg; we can extract operands from the
      // inserted cast
      if constexpr (std::is_same_v<OP, ::imex::dist::LocalTensorOfOp>) {
        rewriter.replaceOp(op, castOp.getInputs()[LTENSOR]);
      } else if constexpr (std::is_same_v<OP, ::imex::dist::TeamOfOp>) {
        rewriter.replaceOp(op, castOp.getInputs()[TEAM]);
      } else if (castOp.getInputs().size() > GSHAPE) {
        // if rank==0 there might be no gshape/loffs args in cast-op
        auto nxt = castOp.getInputs()[GSHAPE];
        auto nxtType = nxt.getType().template dyn_cast<::mlir::MemRefType>();
        if constexpr (std::is_same_v<OP, ::imex::dist::GlobalShapeOfOp>) {
          // Check if this is from block args, e.g. a memref
          if (nxtType) {
            assert(castOp.getInputs().size() == DIST_META_LAST);
            rewriter.replaceOp(
                op, createValuesFromMemRef(rewriter, op.getLoc(),
                                           castOp.getInputs()[GSHAPE]));
          } else {
            // if not a memref, then it originates from a InitTensor
            auto rank = (castOp.getInputs().size() - 2) / 2;
            ::mlir::SmallVector<::mlir::Value, 4> vals(rank);
            for (uint64_t i = 0; i < rank; ++i) {
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
            auto rank = (castOp.getInputs().size() - 2) / 2;
            ::mlir::SmallVector<::mlir::Value, 4> vals(rank);
            for (uint64_t i = 0; i < rank; ++i) {
              vals[i] = castOp.getInputs()[GSHAPE + rank + i];
            }
            rewriter.replaceOp(op, vals);
          }
        } else {
          assert(!"Unknown dist meta item requested");
        }
      } else {
        rewriter.replaceOp(op, ::mlir::ValueRange{});
      }
    }
    return ::mlir::success();
  }
};

using GlobalShapeOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::GlobalShapeOfOp>;
using LocalTensorOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::LocalTensorOfOp>;
using LocalOffsetsOfOpConverter =
    ExtractFromDistOpConverter<::imex::dist::LocalOffsetsOfOp>;
using TeamOfOpConverter = ExtractFromDistOpConverter<::imex::dist::TeamOfOp>;

/// Convert ::imex::dist::LocalPartitionOp into shape and arith calls.
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
    auto loc = op.getLoc();
    auto gShape = adaptor.getGShape();
    int64_t rank = (int64_t)gShape.size();

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

// Compute local slice in dim 0, all other dims are not partitioned (yet)
// return local memref of src
struct LocalOfSliceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::LocalOfSliceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::LocalOfSliceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::LocalOfSliceOp op,
                  ::imex::dist::LocalOfSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto src = op.getDTensor();
    auto inpPtTyp = src.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp)
      return ::mlir::failure();
    auto slcOffs = adaptor.getOffsets();
    auto slcSizes = adaptor.getSizes();
    auto slcStrides = adaptor.getStrides();

    auto zeroIdx = easyIdx(loc, rewriter, 0);
    auto oneIdx = easyIdx(loc, rewriter, 1);

    // Get the local part of the global slice, team, rank, offsets
    int64_t rank = (int64_t)inpPtTyp.getPTensorType().getRank();
    auto lPTnsr = createLocalTensorOf(loc, rewriter, src);
    auto lMemRef = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, inpPtTyp.getPTensorType().getMemRefType(), lPTnsr);
    auto lOffs = createLocalOffsetsOf(loc, rewriter, src);
    auto slcOffs0 = easyIdx(loc, rewriter, slcOffs[0]);
    auto slcSizes0 = easyIdx(loc, rewriter, slcSizes[0]);
    auto slcStrides0 = easyIdx(loc, rewriter, slcStrides[0]);

    // last index of slice
    auto slcEnd = slcOffs0 + slcSizes0 * slcStrides0;
    // local extent/size
    auto lExtent =
        easyIdx(loc, rewriter,
                ::mlir::linalg::createOrFoldDimOp(rewriter, loc, lMemRef, 0));
    // local offset (dim 0)
    auto lOff = easyIdx(loc, rewriter, lOffs[0]);
    // last index of local partition
    auto lEnd = lOff + lExtent;
    // check if requested slice fully before local partition
    auto beforeLocal = slcEnd.ult(lOff);
    // check if requested slice fully behind local partition
    auto behindLocal = lEnd.ule(slcOffs0);
    // check if requested slice start before local partition
    auto startsBefore = slcOffs0.ult(lOff);
    auto strOff = lOff - slcOffs0;
    // (strOff / stride) * stride
    auto nextMultiple = (strOff / slcStrides0) * slcStrides0;
    // Check if local start is on a multiple of the new slice
    auto isMultiple = nextMultiple.eq(strOff);
    // stride - (strOff - nextMultiple)
    auto off = slcStrides0 - (strOff - nextMultiple);
    // offset is either 0 if multiple or off
    auto lDiff1 = isMultiple.select(zeroIdx, off);
    // if view starts within our partition: (start-lOff)
    auto lDiff2 = slcOffs0 - lOff;
    auto viewOff1 = startsBefore.select(lDiff1, lDiff2);
    // except if slice/view before or behind local partition
    auto viewOff0 = beforeLocal.select(zeroIdx, viewOff1);
    // viewOff is the offset from local partition to slice's local start
    auto viewOff = behindLocal.select(zeroIdx, viewOff0);
    // min of lEnd and slice's end
    auto theEnd = lEnd.min(slcEnd);
    // range between local views start and end
    auto lRange = (theEnd - viewOff) - lOff;
    // number of elements in local view (range+stride-1)/stride
    auto viewSize1 = (lRange + (slcStrides0 - oneIdx)) / slcStrides0;
    auto viewSize0 = beforeLocal.select(zeroIdx, viewSize1);
    auto viewSize = behindLocal.select(zeroIdx, viewSize0);
    // start index of local partition's part of slice
    auto gSlcStart1 = (lOff + viewOff - slcOffs0) / slcStrides0;
    auto gSlcStart0 = beforeLocal.select(zeroIdx, gSlcStart1);
    auto gSlcStart = behindLocal.select(zeroIdx, gSlcStart0);

    // and store in output [lOffsets, lSizes, gOffsets]
    ::mlir::SmallVector<::mlir::Value> results(3 * rank);
    results[0 * rank] = viewOff.get();
    results[1 * rank] = viewSize.get();
    results[2 * rank] = gSlcStart.get();
    for (auto i = 1; i < rank; ++i) {
      results[0 * rank + i] = slcOffs[i];
      results[1 * rank + i] = slcSizes[i];
      results[2 * rank + i] = slcOffs[i];
    }

    rewriter.replaceOp(op, results);
    return ::mlir::success();
  }
};

#if 0
  // if requested, also store global offsets
  if(gOffsets) {
    (*gOffsets)[0] = (lOff + viewOff).get();
    for (auto i = 1; i < rank; ++i) {
      (*gOffsets)[i] = slcOffs[i];
    }
  }

  if (doOffs) {
    // offset of local partition in new tensor/view
    auto lViewOff1 = ((lOff + viewOff) - slcOffs0) / slcStrides0;
    auto lViewOff0 = beforeLocal.select(slcOffs0, lViewOff1);
    auto lViewOff = behindLocal.select(slcEnd, lViewOff0);
    // create local offsets from above computed
    auto lViewOffs =
        createAllocMR(rewriter, loc, rewriter.getIndexType(), rank);
    if (rank > 1)
      (void)::mlir::linalg::makeMemRefCopyOp(rewriter, loc, lOffs, lViewOffs);
    rewriter.create<::mlir::memref::StoreOp>(loc, lViewOff.get(), lViewOffs,
                                             zeroIdx.get());

    return lViewOffs;
  }
  return ::mlir::Value();
}
#endif // if 0

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

/// Convert ::imex::dist::ReBalanceOp into runtime call to "_idtr_rebalance".
/// Pass local RankedTensor and result pointer as argument.
/// Replaces op with new DistTensor.
struct ReBalanceOpConverter
    : public ::mlir::OpConversionPattern<::imex::dist::ReBalanceOp> {
  using ::mlir::OpConversionPattern<
      ::imex::dist::ReBalanceOp>::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::dist::ReBalanceOp op,
                  ::imex::dist::ReBalanceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // get guid and rank and call runtime function
    auto loc = op.getLoc();
    auto dTnsr = op.getDTensor().front(); // FIXME
    auto inpPtTyp = dTnsr.getType().dyn_cast<::imex::dist::DistTensorType>();
    if (!inpPtTyp)
      return ::mlir::failure();

    // Get the local part of the global tensor
    auto mRefType = inpPtTyp.getPTensorType().getMemRefType();
    if (mRefType.getRank() == 0) {
      // there is no need to rebalance if 0d -> replace with input
      rewriter.replaceOp(op, dTnsr);
      return ::mlir::success();
    }
    auto lMemRef = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, mRefType, createLocalTensorOf(loc, rewriter, dTnsr));

    auto rank = createIndex(loc, rewriter, mRefType.getRank());
    auto dtype = createDType(loc, rewriter, mRefType);

    auto fsa = rewriter.getStringAttr("_idtr_rebalance");
    // Now get all the input pointers (data, sizes, strides)
    auto meta =
        rewriter.create<::mlir::memref::ExtractStridedMetadataOp>(loc, lMemRef);
    auto dataPtr = createExtractPtrFromMemRef(rewriter, loc, lMemRef, meta);

    auto gShape = createGlobalShapeOf(loc, rewriter, dTnsr);
    auto lOffs = createLocalOffsetsOf(loc, rewriter, dTnsr);
    auto gShapePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, gShape);
    auto lOffsPtr = createExtractPtrFromMemRefFromValues(rewriter, loc, lOffs);
    auto sizePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, meta.getSizes());
    auto stridePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, meta.getStrides());

    // prepare output tensor
    auto team = createTeamOf(loc, rewriter, dTnsr);
    auto nProcs = rewriter.create<::imex::dist::NProcsOp>(loc, team);
    auto pRank = rewriter.create<::imex::dist::PRankOp>(loc, team);
    auto lPart = rewriter.create<::imex::dist::LocalPartitionOp>(loc, nProcs,
                                                                 pRank, gShape);
    auto val = createInt(loc, rewriter, 4711);
    auto outTnsr = rewriter.create<::imex::ptensor::CreateOp>(
        loc, lPart.getLShape(),
        ::imex::ptensor::fromMLIR(mRefType.getElementType()), val, nullptr,
        nullptr); // FIXME device

    auto outMemRef = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, mRefType, outTnsr);
    auto outMeta = rewriter.create<::mlir::memref::ExtractStridedMetadataOp>(
        loc, outMemRef);
    auto outPtr = createExtractPtrFromMemRef(rewriter, loc, outMemRef, outMeta);
    auto outSizePtr =
        createExtractPtrFromMemRefFromValues(rewriter, loc, outMeta.getSizes());
    auto outStridePtr = createExtractPtrFromMemRefFromValues(
        rewriter, loc, outMeta.getStrides());
    auto outRank = createIndex(loc, rewriter, lPart.getLShape().size());
    rewriter.create<::mlir::func::CallOp>(
        loc, fsa, ::mlir::TypeRange(),
        ::mlir::ValueRange{rank, gShapePtr, lOffsPtr, dataPtr, sizePtr,
                           stridePtr, dtype, outRank, outPtr, outSizePtr,
                           outStridePtr});

    rewriter.replaceOp(op, createDistTensor(loc, rewriter, outTnsr, true,
                                            gShape, lOffs, team));
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
          values.resize(off + DIST_META_LAST - (rank ? 0 : 2));
          values[LTENSOR + off] = createLocalTensorOf(loc, builder, value);
          values[TEAM + off] = createTeamOf(loc, builder, value);
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

    // make sure function bouindaries get converted
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

    // No dist should remain
    target.addIllegalDialect<::imex::dist::DistDialect>();
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::imex::ptensor::PTensorDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // All the dist conversion patterns/rewriter
    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<RuntimePrototypesOpConverter, NProcsOpConverter,
                    PRankOpConverter, InitDistTensorOpConverter,
                    LocalPartitionOpConverter, LocalOfSliceOpConverter,
                    GlobalShapeOfOpConverter, LocalTensorOfOpConverter,
                    LocalOffsetsOfOpConverter, TeamOfOpConverter,
                    AllReduceOpConverter, ReBalanceOpConverter>(typeConverter,
                                                                &ctxt);
    // This enables the function boundary handling with the above
    // converters/meterializations
    populateDecomposeCallGraphTypesPatterns(&ctxt, typeConverter, decomposer,
                                            patterns);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
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
