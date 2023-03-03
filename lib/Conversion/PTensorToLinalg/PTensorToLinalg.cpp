//===- PTensorToLinalg.cpp - PTensorToLinalg conversion  -------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PTensorToLinalg conversion, converting the PTensor
/// dialect to the Linalg and helper dialects.
///
/// Any tensor of PTensorType is expected to be initialized by MkPTensorOp.
/// Lowering a MkPtensorOp results in a unrealized_conversion_cast. After
/// complete conversion the resulting value should have no use. However, during
/// conversion its operands will serve for extracting the members (such as
/// ExtractMemRefOp): we chase the unrealized_conversion_cast as the rooting op
/// and return the corresponding operand.
///
/// Currently we do not support propagating device data across function
/// boundaries.
///
/// FIXME: same for device by adding regions.
///
/// The pass is based on a ConversionTarget, TypeConverters, legality checks and
/// conversion patterns.
///
///
//===----------------------------------------------------------------------===//

#include <imex/Conversion/PTensorToLinalg/PTensorToLinalg.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Dialect/PTensor/Transforms/Utils.h>
#include <imex/Utils/ArithUtils.h>

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Func/Transforms/FuncConversions.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Pass/Pass.h>

#include <iostream>

#include "../PassDetail.h"

namespace imex {

/// @return type without a sign
static mlir::Type makeSignlessType(mlir::Type type) {
  if (auto intType = type.dyn_cast<mlir::IntegerType>()) {
    if (!intType.isSignless())
      return mlir::IntegerType::get(intType.getContext(), intType.getWidth());
  }
  return type;
}

/// @return operand cast to signless type if needed, val if not
static mlir::Value doSignCast(mlir::OpBuilder &builder, mlir::Location &loc,
                              mlir::Value val) {
  auto origType = val.getType();
  auto signlessType = makeSignlessType(origType);
  if (signlessType != origType) {
    val =
        builder
            .create<::mlir::UnrealizedConversionCastOp>(loc, signlessType, val)
            .getResult(0);
  }
  return val;
}

/// Create a linalg generic op from given output, input and body
template <typename V, typename B>
auto createParFor(mlir::Location &loc, mlir::OpBuilder &builder, uint64_t rank,
                  ::mlir::Value out, const V &inputs, B bBuilder) {
  // map for output and input
  const ::mlir::AffineMap map =
      ::mlir::AffineMap::getMultiDimIdentityMap(rank, builder.getContext());
  llvm::SmallVector<::mlir::AffineMap> maps(1 + inputs.size(), map);
  llvm::SmallVector<::mlir::utils::IteratorType> iterators(
      rank, mlir::utils::IteratorType::parallel);

  return builder.create<::mlir::linalg::GenericOp>(
      loc, out.getType(), inputs, out, maps, iterators, bBuilder);
}

// *******************************
// ***** Individual patterns *****
// *******************************

namespace {

/// Lower MkPTensorOp into a UnrealizedConversionCastOp, using the type
/// converter to determine the target type. Operations extracting members
/// (tensor, device etc) are expected to chase the tuple creation back to here
/// and get the respective operand of the cast.
// FIXME Is there a better/cleaner way to do this?
// FIXME Right now we simply convert to the tensor, we need proper function
// boundary handling
struct MkPTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::MkPTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::MkPTensorOp op,
                  ::imex::ptensor::MkPTensorOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getTensor());
    return ::mlir::success();
  }
};

/// Lower to the input operand of the defining op. We assume this to ultimately
/// be the UnrealizedConversionCast created by MkPTensorLowering.
struct ExtractTensorLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractMemRefOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractMemRefOp op,
                  ::imex::ptensor::ExtractMemRefOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto inpOp =
        adaptor.getInput().getDefiningOp<::mlir::UnrealizedConversionCastOp>();
    if (!inpOp) { // block arg or similar
      rewriter.replaceOp(op, adaptor.getInput());
    } else {
      // This can be a chain of casts, originating from type conversion like
      // type materialization for function arguments. This requires chasing the
      // chain of casts. We cannot chase casts with more than one operand
      // without getting into realms of unclear semantics.
      while (inpOp && inpOp.getOperands().size() == 1) {
        if (auto defOp =
                inpOp.getOperands()
                    .front()
                    .getDefiningOp<::mlir::UnrealizedConversionCastOp>()) {
          inpOp = defOp;
        } else
          break;
      }
      assert(inpOp);
      assert(inpOp.getOperands().front().getType().isa<::mlir::MemRefType>());
      rewriter.replaceOp(op, inpOp.getOperands()[0]);
    }
    return ::mlir::success();
  }
};

/// Convert PTensor's subview to memref::subview.
/// Adjusted from NTensor
struct SubviewLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::SubviewOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::SubviewOp op,
                  ::imex::ptensor::SubviewOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto src = adaptor.getSource();
    auto srcType = src.getType().dyn_cast<::mlir::MemRefType>();
    if (!srcType)
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType())
                       .dyn_cast_or_null<::mlir::MemRefType>();
    if (!dstType)
      return mlir::failure();

    auto loc = op->getLoc();
    auto offsets = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto sizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                        adaptor.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                          adaptor.getStrides(), rewriter);

    auto resType = ::mlir::memref::SubViewOp::inferRankReducedResultType(
                       dstType.getShape(), srcType, offsets, sizes, strides)
                       .cast<::mlir::MemRefType>();

    mlir::Value res = rewriter.create<::mlir::memref::SubViewOp>(
        loc, resType, src, offsets, sizes, strides);

    assert(resType == dstType);
    // res = rewriter.create<::imex::util::ChangeLayoutOp>(loc, dstType, res);

    rewriter.replaceOp(op, res);

    return ::mlir::success();
  }
};

/// Convert PTensor's LoadOp to memref::LoadOp.
/// Adjusted from NTensor
struct LoadOpLowering
    : public mlir::OpConversionPattern<imex::ptensor::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ptensor::LoadOp op,
                  imex::ptensor::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getArray().getType().cast<imex::ptensor::PTensorType>();
    auto src = adaptor.getArray();
    if (!src.getType().isa<mlir::MemRefType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");

    auto dstType = converter->convertType(op.getType());
    if (!dstType || dstType != origType.getElementType())
      return mlir::failure();

    // auto results = imex::util::wrapEnvRegion(
    //     rewriter, op->getLoc(), origType.getEnvironment(), dstType,
    //     [&](mlir::OpBuilder &builder, mlir::Location loc) {
    rewriter.replaceOpWithNewOp<mlir::memref::LoadOp>(op, src,
                                                      adaptor.getIndices());
    // });

    // rewriter.replaceOp(op, results);
    return mlir::success();
  }
};

/// Convert PTensor's insert_slice to memref
struct InsertSliceLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::InsertSliceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::InsertSliceOp op,
                  ::imex::ptensor::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // get operators
    auto src = adaptor.getSource();
    auto dst = adaptor.getDestination();
    // source and result are expected to be of MemRefType
    auto srcMRTyp = src.getType().dyn_cast<::mlir::MemRefType>();
    auto dstMRTyp = dst.getType().dyn_cast<::mlir::MemRefType>();
    if (!dstMRTyp || !srcMRTyp)
      return ::mlir::failure();

    auto view = rewriter.create<::mlir::memref::SubViewOp>(
        loc, dst, adaptor.getOffsets(), adaptor.getSizes(),
        adaptor.getStrides());
    auto viewMRTyp = view.getType().dyn_cast<::mlir::MemRefType>();
    auto rank = viewMRTyp.getRank();

    // FIXME properly handle broadcasting
    if (srcMRTyp.getRank() == 0) {
      // we just assume the slice size is constant 0 as well
      // assert(getSizeFromValues(adaptor.getSizes()) == 0);
      ::mlir::SmallVector<int64_t> eSz(rank, 1);
      src = rewriter.create<::mlir::memref::ExpandShapeOp>(
          loc, eSz, src, ::llvm::ArrayRef<::mlir::ReassociationIndices>{});
    }

    rewriter.replaceOp(
        op, ::mlir::linalg::makeMemRefCopyOp(rewriter, loc, src, view)
                .getResults());
    return ::mlir::success();
  }
};

#if 0
    auto viewMRTyp = view.getType().dyn_cast<::mlir::MemRefType>();
    auto rank = viewMRTyp.getRank();

    ::std::array<::mlir::AffineMap, 2> maps = {
      ::mlir::AffineMap::getMinorIdentityMap(rank, srcMRTyp.getRank(), op.getContext()),
      ::mlir::AffineMap::getMultiDimIdentityMap(rank, op.getContext())
    };

    // we just make all dims parallel
    ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
        rank, ::mlir::utils::IteratorType::parallel);

    auto srcTnsr = rewriter.create<::mlir::bufferization::ToTensorOp>(loc, src).getResult();
    auto dstTnsr = rewriter.create<::mlir::bufferization::ToTensorOp>(loc, view).getResult();

    // FIXME: make createParFor ready for this
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, dstTnsr.getType(), srcTnsr, dstTnsr, maps, iterators,
        [](::mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args) {
          (void)builder.create<::mlir::linalg::YieldOp>(loc, args[0]);
        });
    // done. replace op with memref
    auto resMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(
        loc, view.getType(), resTnsr.getResult(0));

    // hack to protect linalg.generic from getting erased
    auto z = createIndex(loc, rewriter, 0);
    ::mlir::SmallVector<::mlir::Value> zeros(rank, z);
    rewriter.replaceOpWithNewOp<::mlir::memref::StoreOp>(
        op, rewriter.create<::mlir::memref::LoadOp>(loc, resMR, zeros), resMR, zeros);

    return ::mlir::success();
  }
};
#endif

/// Convert PTensor's arange and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct ARangeLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ARangeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ARangeOp op,
                  ::imex::ptensor::ARangeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // Get Operands
    auto start = easyIdx(loc, rewriter, adaptor.getStart());
    auto stop = easyIdx(loc, rewriter, adaptor.getStop());
    auto step = easyIdx(loc, rewriter, adaptor.getStep());
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!retPtTyp)
      return ::mlir::failure();

    // get arange count
    auto count = createCountARange(rewriter, loc, start, stop, step);

    // init tensor
    auto elTyp = retPtTyp.getElementType();
    auto rank = retPtTyp.getRank();
    auto tensor = createEmptyTensor(rewriter, loc, elTyp, {count});

    // The loop body fills with arange values
    // accepting no input, the lambda simply captures start and step
    auto body = [&start, &step, &elTyp](::mlir::OpBuilder &builder,
                                        ::mlir::Location loc,
                                        ::mlir::ValueRange args) {
      auto dim = getIntAttr<64>(builder, 0);
      auto idx = easyIdx(loc, builder,
                         builder.create<::mlir::linalg::IndexOp>(loc, dim));
      auto val = start + (step * idx);
      // auto _val = builder.create<mlir::arith::SIToFPOp>(loc, elTyp, val);
      (void)builder.create<::mlir::linalg::YieldOp>(
          loc, createIndexCast(loc, builder, val.get(), elTyp));
    };

    auto res =
        createParFor(loc, rewriter, rank, tensor, ::mlir::ValueRange(), body);
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, retPtTyp.getMemRefType(), res.getResult(0));

    return ::mlir::success();
  }
};

/// Convert PTensor's createOp and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct CreateLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::CreateOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::CreateOp op,
                  ::imex::ptensor::CreateOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    // check output type and get operands
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!retPtTyp)
      return ::mlir::failure();
    auto value = adaptor.getValue();

    // init tensor
    auto elTyp = ::imex::ptensor::toMLIR(rewriter, op.getDType());
    auto res = createEmptyTensor(rewriter, loc, elTyp, adaptor.getShape());

    if (value) {
      res = createParFor(
                loc, rewriter, retPtTyp.getRank(), res, ::mlir::ValueRange(),
                [&value](::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::ValueRange args) {
                  (void)builder.create<::mlir::linalg::YieldOp>(loc, value);
                })
                .getResult(0);
    }

    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, getMemRefType(op.getContext(), adaptor.getShape(), elTyp, true),
        res);

    return ::mlir::success();
  }
};

// function type for building body for linalg::generic
using BodyType = std::function<void(
    mlir::OpBuilder &builder, ::mlir::Location loc, ::mlir::ValueRange args)>;

// any genericOp body needs to close with a yield
// we also add a cast op to "typ" if needed
template <typename T>
static void yield(mlir::OpBuilder &builder, ::mlir::Location loc,
                  ::mlir::Type typ, T val) {
  auto res = val;
  if (typ != res.getType()) {
    res = builder.create<::mlir::UnrealizedConversionCastOp>(loc, typ, res)
              .getResult(0);
  }
  (void)builder.create<mlir::linalg::YieldOp>(loc, res);
}

/// Trivial binop builders have simple equivalents in Arith.
/// The Arith ops are accepted as template arguments, one for ints and one for
/// floats. Currently only integers and floats are supported.
/// Currently unsigned int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivial(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto lhsTyp = args[0].getType();
    if (lhsTyp.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        auto lhs = doSignCast(builder, loc, args[0]);
        auto rhs = doSignCast(builder, loc, args[1]);
        yield(builder, loc, typ,
              builder.create<IOP>(loc, lhs, rhs).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (lhsTyp.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ,
              builder.create<FOP>(loc, args[0], args[1]).getResult());
        return;
      } else
        assert(0 && "Found float type but binary op not defined for floats");
    } else {
      assert(0 && "Only integers and floats supported for binary ops");
    }
  };
}

/// get a body builder for given binary operation and result type.
/// Accepts a result type to insert a cast after the operation if needed
/// FIXME: add missing ops
static BodyType getBodyBuilder(::imex::ptensor::EWBinOpId binOp,
                               ::mlir::Type typ) {
  switch (binOp) {
  case ptensor::ADD:
    return buildTrivial<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  // case ptensor::ATAN2] =
  case ptensor::FLOOR_DIVIDE:
    return buildTrivial<mlir::arith::FloorDivSIOp>(typ);
  // case ptensor::LOGADDEXP] =
  // case ptensor::LSHIFT] =
  // case ptensor::MATMUL] =
  case ptensor::MAXIMUM:
    return buildTrivial<mlir::arith::MaxSIOp, mlir::arith::MaxFOp>(typ);
  case ptensor::MINIMUM:
    return buildTrivial<mlir::arith::MinSIOp, mlir::arith::MinFOp>(typ);
  case ptensor::MODULO:
    return buildTrivial<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case ptensor::MULTIPLY:
    return buildTrivial<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  // case ptensor::POW] =
  case ptensor::SUBTRACT:
    return buildTrivial<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  // case ptensor::TRUE_DIVIDE] =
  // case ptensor::BITWISE_AND] =
  // case ptensor::BITWISE_LEFT_SHIFT] =
  // case ptensor::BITWISE_OR] =
  // case ptensor::BITWISE_RIGHT_SHIFT] =
  // case ptensor::BITWISE_XOR] =

  // case ptensor::EQUAL] =
  // case ptensor::GREATER] =
  // case ptensor::GREATER_EQUAL] =
  // case ptensor::LESS] =
  // case ptensor::LESS_EQUAL] =
  // case ptensor::LOGICAL_AND] =
  // case ptensor::LOGICAL_OR] =
  // case ptensor::LOGICAL_XOR] =
  // case ptensor::NOT_EQUAL] =
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

/// Convert PTensor's elementwise binary operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type).
/// Also needs some arith and affine (for linalg::genericop).
struct EWBinOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::EWBinOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWBinOp op,
                  ::imex::ptensor::EWBinOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower PTensors
    auto lhsPtTyp =
        op.getLhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto rhsPtTyp =
        op.getRhs().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!lhsPtTyp || !rhsPtTyp ||
        lhsPtTyp.getMemRefType().getElementType() !=
            rhsPtTyp.getMemRefType().getElementType()) {
      // FIXME type casting
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }
    // we expect tensorType as operands
    auto lhsMRTyp = lhsPtTyp.getMemRefType();
    auto rhsMRTyp = rhsPtTyp.getMemRefType();
    auto elTyp = lhsMRTyp.getElementType();
    auto lhsRank = lhsMRTyp.getRank();
    auto rhsRank = rhsMRTyp.getRank();

    // get the input as tensors
    auto lhsMR = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, lhsMRTyp, op.getLhs());
    auto rhsMR = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, rhsMRTyp, op.getRhs());
    auto lhsTnsr = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, lhsPtTyp.getTensorType(), lhsMR);
    auto rhsTnsr = rewriter.create<::mlir::bufferization::ToTensorOp>(
        loc, rhsPtTyp.getTensorType(), rhsMR);

    // determine broadcasted shape of result
    auto idxType = rewriter.getIndexType();
    auto rank = static_cast<unsigned>(std::max(lhsRank, rhsRank));
    auto lhsShape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, lhsTnsr);
    auto rhsShape = rewriter.create<::mlir::shape::ShapeOfOp>(loc, rhsTnsr);
    auto resShapeType =
        ::mlir::RankedTensorType::get(::std::array<int64_t, 1>{rank}, idxType);
    auto resShape = rewriter.create<::mlir::shape::BroadcastOp>(
        loc, resShapeType, lhsShape, rhsShape, ::mlir::StringAttr{});

    // Init empty result tensor
    llvm::SmallVector<::mlir::Value> resShapeV(rank);
    for (unsigned i = 0; i < rank; ++i) {
      auto idx = createIndex(loc, rewriter, i);
      auto tmp =
          rewriter.createOrFold<::mlir::shape::GetExtentOp>(loc, resShape, idx);
      resShapeV[i] = rewriter.createOrFold<::mlir::shape::SizeToIndexOp>(
          loc, rewriter.getIndexType(), tmp);
    }
    auto tensor = createEmptyTensor(rewriter, loc, elTyp, resShapeV);
    auto resMRTyp = getMemRefType(rewriter.getContext(), rank, elTyp);

    // we need affine maps for linalg::generic
    // as long as we have no proper support for rank-reduced sizes above Linalg,
    // we can handle only
    //   - explicitly rank-reduced inputs (such as explicit 0d tensors)
    //   - shapes with static dim-sizes of 1
    // FIXME: Dynamic dim-sizes of 1 are not properly handled
    ::mlir::SmallVector<::mlir::AffineExpr> lhsExprs, rhsExprs, resExprs;
    for (int i = 0; i < lhsRank; ++i) {
      lhsExprs.emplace_back(lhsMRTyp.getDimSize(i) == 1
                                ? rewriter.getAffineConstantExpr(0)
                                : rewriter.getAffineDimExpr(i));
    }
    for (int i = 0; i < rhsRank; ++i) {
      rhsExprs.emplace_back(rhsMRTyp.getDimSize(i) == 1
                                ? rewriter.getAffineConstantExpr(0)
                                : rewriter.getAffineDimExpr(i));
    }
    for (unsigned i = 0; i < rank; ++i) {
      resExprs.emplace_back(rewriter.getAffineDimExpr(i));
    }
    auto maps =
        ::mlir::AffineMap::inferFromExprList({lhsExprs, rhsExprs, resExprs});

    // we just make all dims parallel
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        rank, ::mlir::utils::IteratorType::parallel);

    // get the body builder for our binop and create genericop
    // FIXME: make createParFor ready for this
    const ::imex::ptensor::EWBinOpId binOpId =
        (::imex::ptensor::EWBinOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, tensor.getType(), ::mlir::ValueRange{lhsTnsr, rhsTnsr}, tensor,
        maps, iterators, bodyBuilder);

    // done. replace op with memref
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, resMRTyp, resTnsr.getResult(0));

    return ::mlir::success();
  }
};

// get a body builder for given binary operation and result type
// we accept a result type to insert a cast after the operation if needed
static BodyType getBodyBuilder(::imex::ptensor::ReduceOpId redOp,
                               ::mlir::Type typ) {
  switch (redOp) {
  case ::imex::ptensor::PROD:
    return getBodyBuilder(::imex::ptensor::MULTIPLY, typ);
  case ::imex::ptensor::SUM:
    return getBodyBuilder(::imex::ptensor::ADD, typ);
  case ::imex::ptensor::MAX:
    return getBodyBuilder(::imex::ptensor::MAXIMUM, typ);
  case ::imex::ptensor::MIN:
    return getBodyBuilder(::imex::ptensor::MINIMUM, typ);
  case ::imex::ptensor::MEAN:
  case ::imex::ptensor::STD:
  case ::imex::ptensor::VAR:
  default:
    assert(0 && "unsupported reduction operation");
  };
}

/// Convert PTensor's reduction operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type). Also needs some arith and affine (for
/// linalg::genericop).
// FIXME reduction over a subset of dimensionsstruct ReductionOpLowering
struct ReductionOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReductionOp op,
                  ::imex::ptensor::ReductionOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower PTensors
    auto inpPtTyp =
        op.getInput().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!inpPtTyp) {
      // fail if not, will be retried if operands get converted elsewhere
      return ::mlir::failure();
    }

    // we expect tensorType as operands
    auto inpTnsrTyp = inpPtTyp.getMemRefType();
    auto inpTnsr = rewriter.create<::imex::ptensor::ExtractMemRefOp>(
        loc, inpTnsrTyp, op.getInput());

    // Get signless operands into vec
    llvm::SmallVector<mlir::Value, 1> oprnds = {
        rewriter.create<::mlir::bufferization::ToTensorOp>(
            loc, inpPtTyp.getTensorType(), inpTnsr)};

    // determine resulting element type from converted op-type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retTyp = retPtTyp.getMemRefType();
    auto elTyp = retTyp.getElementType();
    auto sElTyp = makeSignlessType(elTyp);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(retTyp.getRank());
    assert(rank == 0);
    auto zeroI = createIndex(loc, rewriter, 0);
    llvm::SmallVector<::mlir::Value> shapeVVec(rank, zeroI);
    // create new tensor
    auto zero = createInt(loc, rewriter, 0);
    auto tensor = createEmptyTensor(rewriter, loc, sElTyp, shapeVVec);
    auto tnsr = rewriter.create<::mlir::linalg::FillOp>(loc, zero, tensor);

    // rank/num-dims of input
    auto inpRank = static_cast<unsigned>(inpTnsrTyp.getRank());
    // input maps are identity maps
    auto inpMap = ::mlir::AffineMap::getMultiDimIdentityMap(
        inpRank, rewriter.getContext());
    // output map is "*->()"
    auto omap = ::mlir::AffineMap::get(inpRank, 0, rewriter.getContext());
    const ::mlir::AffineMap maps[] = {inpMap, omap};
    llvm::SmallVector<mlir::utils::IteratorType> iterators(
        inpRank, mlir::utils::IteratorType::reduction);

    // create reduction op as linalg::generic
    const ::imex::ptensor::ReduceOpId ropid =
        (::imex::ptensor::ReduceOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();
    auto bodyBuilder = getBodyBuilder(ropid, sElTyp);
    auto resTnsr = rewriter.create<::mlir::linalg::GenericOp>(
        loc, tnsr.getType(0), oprnds, tnsr.getResult(0), maps, iterators,
        bodyBuilder);
    rewriter.replaceOpWithNewOp<::mlir::bufferization::ToMemrefOp>(
        op, retPtTyp.getMemRefType(), resTnsr.getResult(0));

    return ::mlir::success();
  }
};

// *******************************
// ***** Pass infrastructure *****
// *******************************

/// Convert PTensor to Linalg.
/// After success, no more PTensor should be left, replaced by Linalg & Affine &
/// Arith. Use a type converter to get rid of PTensorType.
struct ConvertPTensorToLinalgPass
    : public ::imex::ConvertPTensorToLinalgBase<ConvertPTensorToLinalgPass> {

  ConvertPTensorToLinalgPass() = default;

  void runOnOperation() override {
    auto &ctxt = getContext();
    ::mlir::ConversionTarget target(ctxt);
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // Convert PTensorType to (tensorType, device, team, handle)
    auto convPt2Rt = [&ctxt](::imex::ptensor::PTensorType type)
        -> ::mlir::Optional<::mlir::Type> { return type.getMemRefType(); };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convPt2Rt);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // We convert all PTensor stuff...
    target.addIllegalDialect<::imex::ptensor::PTensorDialect>();
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<::mlir::linalg::LinalgDialect>();
    target.addLegalDialect<::mlir::AffineDialect>();
    target.addLegalDialect<::mlir::arith::ArithDialect>();
    target.addLegalDialect<::mlir::memref::MemRefDialect>();
    target.addLegalDialect<::mlir::tensor::TensorDialect>();
    target.addLegalDialect<::mlir::shape::ShapeDialect>();
    target.addLegalDialect<::mlir::bufferization::BufferizationDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME
    // make sure function boundaries use tensors (not PTensors)
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp>(
        [&](::mlir::func::ReturnOp op) { return typeConverter.isLegal(op); });
    target.addDynamicallyLegalOp<mlir::func::CallOp>(
        [&](mlir::func::CallOp op) { return typeConverter.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns.insert<MkPTensorLowering, ExtractTensorLowering, SubviewLowering,
                    InsertSliceLowering, ARangeLowering, CreateLowering,
                    EWBinOpLowering, ReductionOpLowering>(typeConverter, &ctxt);
    ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
        ::mlir::func::FuncOp>(patterns, typeConverter);
    ::mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter);
    ::mlir::populateCallOpTypeConversionPattern(patterns, typeConverter);

    if (::mlir::failed(::mlir::applyPartialConversion(getOperation(), target,
                                                      ::std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

/// Create a pass to convert PTensor to Linalg
std::unique_ptr<::mlir::OperationPass<::mlir::ModuleOp>>
createConvertPTensorToLinalgPass() {
  return std::make_unique<ConvertPTensorToLinalgPass>();
}

} // namespace imex
