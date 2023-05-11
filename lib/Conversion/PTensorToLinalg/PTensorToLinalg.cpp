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
/// ExtractTensorOp): we chase the unrealized_conversion_cast as the rooting op
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
#include <mlir/Dialect/Math/IR/Math.h>
#include <mlir/Dialect/SCF/Transforms/Transforms.h>
#include <mlir/Dialect/Shape/IR/Shape.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
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
  ::mlir::SmallVector<::mlir::AffineMap> maps(1 + inputs.size(), map);
  ::mlir::SmallVector<::mlir::utils::IteratorType> iterators(
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
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractTensorOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractTensorOp op,
                  ::imex::ptensor::ExtractTensorOp::Adaptor adaptor,
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
      // std::cerr << "mrOpOp: ";
      // inpOp->getOperand(0).getDefiningOp()->getOperand(0).dump(); std::cerr
      // << "mrOp: "; inpOp->getOperands().front().dump(); std::cerr << "mr: ";
      // inpOp->dump();
      rewriter.replaceOp(op, inpOp.getOperands()[0]);
    }
    return ::mlir::success();
  }
};

/// Lower to the input operand of the defining op to a raw pointer.
struct ExtractRawPtrLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ExtractRawPtrOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ExtractRawPtrOp op,
                  ::imex::ptensor::ExtractRawPtrOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    auto loc = op->getLoc();
    auto srcTnsr = adaptor.getSource();
    auto srcType = srcTnsr.getType().dyn_cast<::mlir::TensorType>();
    if (!srcType) {
      return mlir::failure();
    }

    // convert src tensor to memref
    auto srcPtType = op.getSource()
                         .getType()
                         .dyn_cast_or_null<::imex::ptensor::PTensorType>();
    if (!srcPtType)
      return mlir::failure();
    auto srcMRType = srcPtType.getMemRefType();
    auto srcMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(
        loc, srcMRType, srcTnsr);

    rewriter.replaceOp(
        op, createExtractPtrFromMemRef(rewriter, op.getLoc(), srcMR));

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

    auto srcTnsr = adaptor.getSource();
    auto loc = op->getLoc();

    // convert src tensor to memref
    auto srcPtType = op.getSource()
                         .getType()
                         .dyn_cast_or_null<::imex::ptensor::PTensorType>();
    if (!srcPtType)
      return mlir::failure();
    auto srcMRType = srcPtType.getMemRefType();
    auto srcMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(
        loc, srcMRType, srcTnsr);

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");
    auto dstTnsrType = converter->convertType(op.getType())
                           .dyn_cast_or_null<::mlir::TensorType>();
    if (!dstTnsrType)
      return mlir::failure();

    auto offsets = ::mlir::getMixedValues(adaptor.getStaticOffsets(),
                                          adaptor.getOffsets(), rewriter);
    auto sizes = ::mlir::getMixedValues(adaptor.getStaticSizes(),
                                        adaptor.getSizes(), rewriter);
    auto strides = ::mlir::getMixedValues(adaptor.getStaticStrides(),
                                          adaptor.getStrides(), rewriter);

    auto resType =
        ::mlir::memref::SubViewOp::inferRankReducedResultType(
            dstTnsrType.getShape(), srcMRType, offsets, sizes, strides)
            .cast<::mlir::MemRefType>();

    auto sw = rewriter.create<::mlir::memref::SubViewOp>(
        loc, resType, srcMR, offsets, sizes, strides);

    assert(resType.getShape() == dstTnsrType.getShape());

    // convert result to tensor
    auto res = rewriter.create<::mlir::bufferization::ToTensorOp>(loc, sw,
                                                                  false, true);
    rewriter.replaceOp(op, res.getResult());

    return ::mlir::success();
  }
};

/// Convert PTensor's LoadOp to tensor::ExtractOp.
/// Adjusted from NTensor
struct LoadOpLowering
    : public mlir::OpConversionPattern<imex::ptensor::LoadOp> {
  using OpConversionPattern::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(imex::ptensor::LoadOp op,
                  imex::ptensor::LoadOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto srcPtType = op.getArray().getType().cast<imex::ptensor::PTensorType>();
    auto srcTnsr = adaptor.getArray();
    if (!srcTnsr.getType().isa<mlir::TensorType>())
      return mlir::failure();

    auto *converter = getTypeConverter();
    assert(converter && "Type converter is not set");
    auto dstType = converter->convertType(op.getType());
    if (!dstType || dstType != srcPtType.getElementType())
      return mlir::failure();

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractOp>(op, srcTnsr,
                                                         adaptor.getIndices());

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
    auto srcTyp = src.getType().dyn_cast<::mlir::TensorType>();
    auto dstTyp = dst.getType().dyn_cast<::mlir::TensorType>();
    if (!dstTyp || !srcTyp)
      return ::mlir::failure();

    auto srcPtType =
        op.getSource().getType().cast<imex::ptensor::PTensorType>();
    auto dstPtType =
        op.getDestination().getType().cast<imex::ptensor::PTensorType>();

    auto srcMRTyp = srcPtType.getMemRefType();
    auto dstMRTyp = dstPtType.getMemRefType();
    mlir::Value srcMR =
        rewriter.create<::mlir::bufferization::ToMemrefOp>(loc, srcMRTyp, src);
    auto dstMR =
        rewriter.create<::mlir::bufferization::ToMemrefOp>(loc, dstMRTyp, dst);

    auto rank = dstMRTyp.getRank();
    auto slcOffs = adaptor.getOffsets();
    auto slcSizes = adaptor.getSizes();
    auto slcStrides = adaptor.getStrides();

#if HAVE_KDYNAMIC_SIZED_OPS
    ::mlir::SmallVector<::mlir::Value> szs;
    for (auto i = 0; i < rank; ++i) {
      // results[0 * rank + i] = slcOffs[i];
      if (auto cval = ::mlir::getConstantIntValue(slcSizes[i]);
          cval && cval == ::mlir::ShapedType::kDynamic) {
        if (auto oval = ::mlir::getConstantIntValue(slcOffs[i]);
            oval && oval == 0) {
          if (auto sval = ::mlir::getConstantIntValue(slcStrides[i]);
              sval && sval == 1) {
            szs.emplace_back(
                ::mlir::linalg::createOrFoldDimOp(rewriter, loc, dstMR, i));
            continue;
          }
        }
        assert(!"Unspecified end in slice implemented only if slice is "
                "equivalent to '0::1'");
      } else {
        szs.emplace_back(slcSizes[i]);
      }
    }
    auto view = rewriter.create<::mlir::memref::SubViewOp>(loc, dstMR, slcOffs,
                                                           szs, slcStrides);
#else  // HAVE_KDYNAMIC_SIZED_OPS
    auto view = rewriter.create<::mlir::memref::SubViewOp>(
        loc, dstMR, slcOffs, slcSizes, slcStrides);
#endif // HAVE_KDYNAMIC_SIZED_OPS

    // FIXME properly handle broadcasting
    if (srcMRTyp.getRank() == 0) {
      ::mlir::SmallVector<int64_t> eSz(rank, 1);
      srcMR = rewriter.create<::mlir::memref::ExpandShapeOp>(
          loc, eSz, srcMR, ::llvm::ArrayRef<::mlir::ReassociationIndices>{});
    }

    auto sz =
        easyIdx(loc, rewriter,
                ::mlir::linalg::createOrFoldDimOp(rewriter, loc, view, 0));
    for (int64_t i = 1; i < rank; ++i) {
      sz = sz *
           easyIdx(loc, rewriter,
                   ::mlir::linalg::createOrFoldDimOp(rewriter, loc, view, i));
    }
    auto gt0 = rewriter.create<::mlir::arith::CmpIOp>(
        loc, ::mlir::arith::CmpIPredicate::sgt, sz.get(),
        createIndex(loc, rewriter, 0));

    rewriter.replaceOpWithNewOp<::mlir::scf::IfOp>(
        op, gt0,
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          builder.create<::mlir::scf::YieldOp>(
              loc, ::mlir::linalg::makeMemRefCopyOp(builder, loc, srcMR, view)
                       .getResults());
        },
        [&](::mlir::OpBuilder &builder, ::mlir::Location loc) {
          builder.create<::mlir::scf::YieldOp>(loc, ::mlir::ValueRange{});
        });

    return ::mlir::success();
  }
};

/// Convert PTensor's linspace and its return type to Linalg/tensor.
/// Also needs some arith and affine (for linalg::genericop).
struct LinSpaceLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::LinSpaceOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::LinSpaceOp op,
                  ::imex::ptensor::LinSpaceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto start = adaptor.getStart();
    auto stop = adaptor.getStop();
    auto count = adaptor.getNum();
    bool endpoint = adaptor.getEndpoint();
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();

    if (!(start.getType().isIntOrIndexOrFloat() &&
          stop.getType().isIntOrIndexOrFloat() &&
          count.getType().isIntOrIndex() && retPtTyp)) {
      return ::mlir::failure();
    } // FIXME type promotion

    // cast types and get step
    auto f64Type = rewriter.getF64Type();
    count = createIndexCast(loc, rewriter, count);
    start = createCast(loc, rewriter, start, f64Type);
    stop = createCast(loc, rewriter, stop, f64Type);
    auto step = createStepLinSpace(rewriter, loc, start, stop, count, endpoint);

    // init tensor
    auto rank = retPtTyp.getRank();
    auto elTyp = retPtTyp.getElementType();
    auto tensor = createEmptyTensor(rewriter, loc, elTyp, {count});

    // The loop body fills with values
    // accepting no input, the lambda simply captures start and step
    auto body = [&](::mlir::OpBuilder &builder, ::mlir::Location loc,
                    ::mlir::ValueRange args) {
      auto dim = getIntAttr<64>(builder, 0);
      auto idx = createCast(loc, builder,
                            builder.create<::mlir::linalg::IndexOp>(loc, dim),
                            f64Type);
      ::mlir::Value val = builder.create<::mlir::arith::AddFOp>(
          loc, builder.create<::mlir::arith::MulFOp>(loc, step, idx), start);
      (void)builder.create<::mlir::linalg::YieldOp>(
          loc, createCast(loc, rewriter, val, elTyp));
    };

    auto res =
        createParFor(loc, rewriter, rank, tensor, ::mlir::ValueRange(), body);
    rewriter.replaceOp(op, res.getResult(0));

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
    ::mlir::Value res =
        createEmptyTensor(rewriter, loc, elTyp, adaptor.getShape());

    if (value) {
      res = createParFor(
                loc, rewriter, retPtTyp.getRank(), res, ::mlir::ValueRange(),
                [&value](::mlir::OpBuilder &builder, ::mlir::Location loc,
                         ::mlir::ValueRange args) {
                  (void)builder.create<::mlir::linalg::YieldOp>(loc, value);
                })
                .getResult(0);
    }
    rewriter.replaceOp(op, res);

    return ::mlir::success();
  }
};

/// Convert PTensor's ReshapeOp and its return type to Linalg/tensor.
/// Optionally creates a copy first.
struct ReshapeLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::ReshapeOp op,
                  ::imex::ptensor::ReshapeOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // check output type and get operands
    auto retPtTyp = op.getType().dyn_cast<::imex::ptensor::PTensorType>();
    auto srcPtTyp =
        op.getSrc().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!(retPtTyp && srcPtTyp)) {
      return ::mlir::failure();
    }

    auto loc = op.getLoc();
    auto src = adaptor.getSrc();
    auto srcTnsr = src.getType().cast<::mlir::TensorType>();
    auto shape = adaptor.getShape();
    auto elTyp = srcTnsr.getElementType();
    auto outTyp = retPtTyp.getTensorType();

    if (adaptor.getCopy().value_or(false)) {
      auto rank = srcTnsr.getRank();
      ::mlir::SmallVector<::mlir::Value> dims(rank);
      for (int64_t i = 0; i < rank; ++i) {
        dims[i] = ::mlir::linalg::createOrFoldDimOp(rewriter, loc, src, i);
      }
      auto cpy = createEmptyTensor(rewriter, loc, elTyp, dims);
      auto cpyMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(
          loc, getMemRefType(op.getContext(), rank, elTyp, false), cpy);
      auto srcMR = rewriter.create<::mlir::bufferization::ToMemrefOp>(
          loc, srcPtTyp.getMemRefType(), src);
      rewriter.create<::mlir::memref::CopyOp>(loc, srcMR, cpyMR);
      src = cpy;
    }

    auto shapeT = rewriter.create<::mlir::tensor::FromElementsOp>(loc, shape);
    rewriter.replaceOpWithNewOp<::mlir::tensor::ReshapeOp>(op, outTyp, src,
                                                           shapeT);

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
static BodyType buildTrivialBinary(::mlir::Type typ) {
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

/// Trivial unary op builders have simple equivalents in Math.
/// The Math ops are accepted as template arguments, one for ints and one for
/// floats. Currently only integers and floats are supported.
/// Currently unsigned int ops are not supported.
template <typename IOP, typename FOP = void>
static BodyType buildTrivialUnary(::mlir::Type typ) {
  return [typ](mlir::OpBuilder &builder, ::mlir::Location loc,
               ::mlir::ValueRange args) -> void {
    auto srcTyp = args[0].getType();
    if (srcTyp.isIntOrIndex()) {
      if constexpr (!std::is_same_v<IOP, void>) {
        auto src = doSignCast(builder, loc, args[0]);
        yield(builder, loc, typ, builder.create<IOP>(loc, src).getResult());
        return;
      } else
        assert(0 &&
               "Found integer type but binary op not defined for integers");
    } else if (srcTyp.isIntOrIndexOrFloat()) {
      if constexpr (!std::is_same_v<FOP, void>) {
        yield(builder, loc, typ, builder.create<FOP>(loc, args[0]).getResult());
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
    return buildTrivialBinary<mlir::arith::AddIOp, mlir::arith::AddFOp>(typ);
  case ptensor::ATAN2:
    return buildTrivialBinary<void, mlir::math::Atan2Op>(typ);
  case ptensor::FLOOR_DIVIDE:
    return buildTrivialBinary<mlir::arith::FloorDivSIOp>(typ);
  // case ptensor::LOGADDEXP] =
  // case ptensor::MATMUL] =
  case ptensor::MAXIMUM:
    return buildTrivialBinary<mlir::arith::MaxSIOp, mlir::arith::MaxFOp>(typ);
  case ptensor::MINIMUM:
    return buildTrivialBinary<mlir::arith::MinSIOp, mlir::arith::MinFOp>(typ);
  case ptensor::MODULO:
    return buildTrivialBinary<mlir::arith::RemSIOp, mlir::arith::RemFOp>(typ);
  case ptensor::MULTIPLY:
    return buildTrivialBinary<mlir::arith::MulIOp, mlir::arith::MulFOp>(typ);
  case ptensor::POWER:
    return buildTrivialBinary<mlir::math::IPowIOp, mlir::math::PowFOp>(typ);
  case ptensor::SUBTRACT:
    return buildTrivialBinary<mlir::arith::SubIOp, mlir::arith::SubFOp>(typ);
  case ptensor::TRUE_DIVIDE:
    return buildTrivialBinary<::mlir::arith::DivSIOp, ::mlir::arith::DivFOp>(
        typ);
  // case ptensor::BITWISE_LEFT_SHIFT] =
  // case ptensor::BITWISE_RIGHT_SHIFT] =

  // case ptensor::EQUAL] =
  // case ptensor::GREATER] =
  // case ptensor::GREATER_EQUAL] =
  // case ptensor::LESS] =
  // case ptensor::LESS_EQUAL] =
  // case ptensor::NOT_EQUAL] =
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

::mlir::Value createTosaOp(::mlir::Location loc,
                           ::imex::ptensor::EWBinOpId binOpId,
                           ::mlir::ConversionPatternRewriter &rewriter,
                           ::mlir::TensorType returnType, ::mlir::Value lhs,
                           ::mlir::Value rhs) {
  switch (binOpId) {
  case ptensor::BITWISE_AND:
    return rewriter
        .create<::mlir::tosa::BitwiseAndOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ptensor::BITWISE_OR:
    return rewriter.create<::mlir::tosa::BitwiseOrOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ptensor::BITWISE_XOR:
    return rewriter
        .create<::mlir::tosa::BitwiseXorOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ptensor::LOGICAL_AND:
    return rewriter
        .create<::mlir::tosa::LogicalAndOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ptensor::LOGICAL_OR:
    return rewriter.create<::mlir::tosa::LogicalOrOp>(loc, returnType, lhs, rhs)
        .getResult();
  case ptensor::LOGICAL_XOR:
    return rewriter
        .create<::mlir::tosa::LogicalXorOp>(loc, returnType, lhs, rhs)
        .getResult();
  default:
    break;
  };
  return ::mlir::Value();
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

    auto resType = op->getResult(0)
                       .getType()
                       .cast<::imex::ptensor::PTensorType>()
                       .getTensorType();

    // get the input as tensors
    auto lhs = adaptor.getLhs();
    auto rhs = adaptor.getRhs();
    auto lhsTnsr = lhs.getType().cast<::mlir::TensorType>();
    auto rhsTnsr = rhs.getType().cast<::mlir::TensorType>();

    // we expect tensorType as operands
    auto elTyp = lhsTnsr.getElementType();
    auto lhsRank = lhsTnsr.getRank();
    auto rhsRank = rhsTnsr.getRank();

    auto rank = static_cast<unsigned>(std::max(lhsRank, rhsRank));

    const ::imex::ptensor::EWBinOpId binOpId =
        (::imex::ptensor::EWBinOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();

    ::mlir::Value newOp =
        createTosaOp(loc, binOpId, rewriter, resType, lhs, rhs);
    if (!newOp) {
      // generate linalg.generic loop

      // create output tensor with right dimensions
      auto tensor = createEmptyTensor(rewriter, loc, resType, {lhs, rhs});

      // we need affine maps for linalg::generic
      // as long as we have no proper support for rank-reduced sizes above
      // Linalg, we can handle only
      //   - explicitly rank-reduced inputs (such as explicit 0d tensors)
      //   - shapes with static dim-sizes of 1
      ::mlir::SmallVector<::mlir::AffineExpr> lhsExprs, rhsExprs, resExprs;
      for (int i = 0; i < lhsRank; ++i) {
        lhsExprs.emplace_back(lhsTnsr.getDimSize(i) == 1
                                  ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(i));
      }
      for (int i = 0; i < rhsRank; ++i) {
        rhsExprs.emplace_back(rhsTnsr.getDimSize(i) == 1
                                  ? rewriter.getAffineConstantExpr(0)
                                  : rewriter.getAffineDimExpr(i));
      }
      for (unsigned i = 0; i < rank; ++i) {
        resExprs.emplace_back(rewriter.getAffineDimExpr(i));
      }
      auto lhsMap = ::mlir::AffineMap::get(resType.getRank(), /*symbolCount=*/0,
                                           lhsExprs, rewriter.getContext());
      auto rhsMap = ::mlir::AffineMap::get(resType.getRank(), /*symbolCount=*/0,
                                           rhsExprs, rewriter.getContext());
      auto resMap = rewriter.getMultiDimIdentityMap(resType.getRank());

      // we just make all dims parallel
      ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
          rank, ::mlir::utils::IteratorType::parallel);

      // get the body builder for our binop and create genericop
      // FIXME: make createParFor ready for this
      auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
      newOp =
          rewriter
              .create<::mlir::linalg::GenericOp>(
                  loc, tensor.getType(), ::mlir::ValueRange{lhs, rhs}, tensor,
                  ::mlir::ArrayRef<::mlir::AffineMap>{lhsMap, rhsMap, resMap},
                  iterators, bodyBuilder)
              .getResult(0);
    }
    rewriter.replaceOp(op, newOp);

    return ::mlir::success();
  }
};

/// get a body builder for given binary operation and result type.
/// Accepts a result type to insert a cast after the operation if needed
/// FIXME: add missing ops
static BodyType getBodyBuilder(::imex::ptensor::EWUnyOpId binOp,
                               ::mlir::Type typ) {
  switch (binOp) {
  case ptensor::ABS:
    return buildTrivialUnary<::mlir::math::AbsIOp, ::mlir::math::AbsFOp>(typ);
  case ptensor::ATAN:
    return buildTrivialUnary<void, ::mlir::math::AtanOp>(typ);
  case ptensor::CEIL:
    return buildTrivialUnary<void, ::mlir::math::CeilOp>(typ);
  case ptensor::COS:
    return buildTrivialUnary<void, ::mlir::math::CosOp>(typ);
  case ptensor::ERF:
    return buildTrivialUnary<void, ::mlir::math::ErfOp>(typ);
  case ptensor::EXP:
    return buildTrivialUnary<void, ::mlir::math::ExpOp>(typ);
  case ptensor::EXPM1:
    return buildTrivialUnary<void, ::mlir::math::ExpM1Op>(typ);
  case ptensor::FLOOR:
    return buildTrivialUnary<void, ::mlir::math::FloorOp>(typ);
  case ptensor::LOG:
    return buildTrivialUnary<void, ::mlir::math::LogOp>(typ);
  case ptensor::LOG1P:
    return buildTrivialUnary<void, ::mlir::math::Log1pOp>(typ);
  case ptensor::LOG2:
    return buildTrivialUnary<void, ::mlir::math::Log2Op>(typ);
  case ptensor::LOG10:
    return buildTrivialUnary<void, ::mlir::math::Log10Op>(typ);
  case ptensor::ROUND:
    return buildTrivialUnary<void, ::mlir::math::RoundOp>(typ);
  case ptensor::SIN:
    return buildTrivialUnary<void, ::mlir::math::SinOp>(typ);
  case ptensor::SQRT:
    return buildTrivialUnary<void, ::mlir::math::SqrtOp>(typ);
  case ptensor::TAN:
    return buildTrivialUnary<void, ::mlir::math::TanOp>(typ);
  case ptensor::TANH:
    return buildTrivialUnary<void, ::mlir::math::TanhOp>(typ);
  case ptensor::TRUNC:
    return buildTrivialUnary<void, ::mlir::math::TruncOp>(typ);
  default:
    assert(0 && "unsupported elementwise binary operation");
  };
}

::mlir::Value createUnaryTosaOp(::mlir::Location loc,
                                ::imex::ptensor::EWUnyOpId binOpId,
                                ::mlir::ConversionPatternRewriter &rewriter,
                                ::mlir::TensorType returnType,
                                ::mlir::Value src) {
  switch (binOpId) {
  case ptensor::LOGICAL_NOT:
    return rewriter.create<mlir::tosa::LogicalNotOp>(loc, returnType, src)
        .getResult();
  default:
    break;
  };
  return ::mlir::Value();
}

/// Convert PTensor's elementwise unary operations and their return type to
/// Linalg/tensor. The given op's type is expected to convert to the appropriate
/// type (shape and element-type).
/// Also needs some arith and affine (for linalg::genericop).
struct EWUnyOpLowering
    : public ::mlir::OpConversionPattern<::imex::ptensor::EWUnyOp> {
  using OpConversionPattern::OpConversionPattern;

  ::mlir::LogicalResult
  matchAndRewrite(::imex::ptensor::EWUnyOp op,
                  ::imex::ptensor::EWUnyOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // We expect to lower PTensors
    auto srcPtTyp =
        op.getSrc().getType().dyn_cast<::imex::ptensor::PTensorType>();
    if (!srcPtTyp) {
      // FIXME type casting
      return ::mlir::failure();
    }

    auto resType = op->getResult(0)
                       .getType()
                       .cast<::imex::ptensor::PTensorType>()
                       .getTensorType();

    // get the input as tensors
    auto src = adaptor.getSrc();
    auto srcTnsr = src.getType().cast<::mlir::TensorType>();

    // we expect tensorType as operands
    auto elTyp = srcTnsr.getElementType();
    auto rank = srcTnsr.getRank();

    const ::imex::ptensor::EWUnyOpId binOpId =
        (::imex::ptensor::EWUnyOpId)adaptor.getOp()
            .cast<::mlir::IntegerAttr>()
            .getInt();

    ::mlir::Value newOp =
        createUnaryTosaOp(loc, binOpId, rewriter, resType, src);

    if (!newOp) {
      // generate linalg.generic loop

      // create output tensor with right dimensions
      auto tensor = createEmptyTensor(rewriter, loc, resType, {src});

      // we need affine maps for linalg::generic
      const ::mlir::AffineMap map = ::mlir::AffineMap::getMultiDimIdentityMap(
          rank, rewriter.getContext());
      ::mlir::SmallVector<::mlir::AffineMap> maps(2, map);
      // we just make all dims parallel
      ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
          rank, ::mlir::utils::IteratorType::parallel);

      // get the body builder for our binop and create genericop
      // FIXME: make createParFor ready for this
      auto bodyBuilder = getBodyBuilder(binOpId, elTyp);
      newOp = rewriter
                  .create<::mlir::linalg::GenericOp>(
                      loc, tensor.getType(), ::mlir::ValueRange{src}, tensor,
                      maps, iterators, bodyBuilder)
                  .getResult(0);
    }

    rewriter.replaceOp(op, newOp);

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
    auto inpTnsr = adaptor.getInput();
    auto inpTnsrTyp = inpTnsr.getType().cast<::mlir::TensorType>();

    // Get signless operands into vec
    ::mlir::SmallVector<mlir::Value, 1> oprnds = {inpTnsr};

    // determine resulting element type from converted op-type
    auto retPtTyp =
        op.getResult().getType().dyn_cast<::imex::ptensor::PTensorType>();
    assert(retPtTyp);
    auto retTyp = retPtTyp.getTensorType();
    auto elTyp = retTyp.getElementType();
    auto sElTyp = makeSignlessType(elTyp);

    // build tensor using the resulting element type and shape
    // FIXME support reduction dimensions
    auto rank = static_cast<unsigned>(retTyp.getRank());
    assert(rank == 0);
    auto zeroI = createIndex(loc, rewriter, 0);
    ::mlir::SmallVector<::mlir::Value> shapeVVec(rank, zeroI);
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
    ::mlir::SmallVector<mlir::utils::IteratorType> iterators(
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
    rewriter.replaceOp(op, resTnsr.getResult(0));

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
    ::mlir::TypeConverter typeConverter;
    // Convert unknown types to itself
    auto convT2T = [](::mlir::Type type) { return type; };
    // Convert PTensorType to (tensorType, device, team, handle)
    auto convPt2Rt = [&ctxt](::imex::ptensor::PTensorType type)
        -> ::mlir::Optional<::mlir::Type> { return type.getTensorType(); };

    typeConverter.addConversion(convT2T);
    typeConverter.addConversion(convPt2Rt);

    auto materializeCast =
        [](::mlir::OpBuilder &builder, ::mlir::Type type,
           ::mlir::ValueRange inputs,
           ::mlir::Location loc) -> ::llvm::Optional<::mlir::Value> {
      if (inputs.size() == 1) {
        auto input = inputs[0];
        if (type.isa<::mlir::TensorType>() and
            input.getType().isa<::mlir::TensorType>()) {
          return builder.create<::mlir::tensor::CastOp>(loc, type, inputs)
              .getResult();
        }
        auto ttype = input.getType().dyn_cast<::mlir::RankedTensorType>();
        if (ttype && type.isa<::mlir::MemRefType>()) {
          return builder
              .create<::mlir::bufferization::ToMemrefOp>(loc, type, input)
              .getResult();
        }
        auto mrtype = input.getType().dyn_cast<::mlir::MemRefType>();
        if (mrtype && type.isa<::mlir::RankedTensorType>()) {
          return builder
              .create<::mlir::bufferization::ToTensorOp>(loc, type, input)
              .getResult();
        }
      }
      return builder
          .create<::mlir::UnrealizedConversionCastOp>(loc, type, inputs)
          .getResult(0);
    };
    typeConverter.addSourceMaterialization(materializeCast);
    typeConverter.addTargetMaterialization(materializeCast);

    // At function boundaries we have actual memref semantics.
    // We need to explicitly convert in/out arguments to memrefs.
    // If we use tensors downstream passes will auto-convert to non-strided
    // memrefs which will imply a copy (converting from strided to non-strided
    // requires a copy)
    // We simply use a separate type-converter and materializations

    ::mlir::TypeConverter typeConverter4Func;
    // Convert PTensorType to MemRefType
    auto convPt2MR = [&ctxt](::imex::ptensor::PTensorType type)
        -> ::mlir::Optional<::mlir::Type> { return type.getMemRefType(); };

    typeConverter4Func.addConversion(convT2T);
    typeConverter4Func.addConversion(convPt2MR);
    typeConverter4Func.addSourceMaterialization(materializeCast);
    typeConverter4Func.addTargetMaterialization(materializeCast);

    ::mlir::ConversionTarget target(ctxt);
    // We convert all PTensor stuff...
    target.addIllegalDialect<::imex::ptensor::PTensorDialect>();
    // ...into Linalg, Affine, Tensor, Arith
    target.addLegalDialect<
        ::mlir::linalg::LinalgDialect, ::mlir::AffineDialect,
        ::mlir::arith::ArithDialect, ::mlir::math::MathDialect,
        ::mlir::memref::MemRefDialect, ::mlir::tensor::TensorDialect,
        ::mlir::tosa::TosaDialect, ::mlir::shape::ShapeDialect,
        ::mlir::bufferization::BufferizationDialect>();
    target.addLegalOp<::mlir::UnrealizedConversionCastOp>(); // FIXME

    // make sure function boundaries use tensors (not PTensors)
    target.addDynamicallyLegalOp<::mlir::func::FuncOp>(
        [&](::mlir::func::FuncOp op) {
          return typeConverter4Func.isSignatureLegal(op.getFunctionType()) &&
                 typeConverter4Func.isLegal(&op.getBody());
        });
    target.addDynamicallyLegalOp<::mlir::func::ReturnOp, mlir::func::CallOp>(
        [&](mlir::Operation *op) { return typeConverter4Func.isLegal(op); });

    ::mlir::RewritePatternSet patterns(&ctxt);
    patterns
        .insert<MkPTensorLowering, ExtractTensorLowering, ExtractRawPtrLowering,
                SubviewLowering, InsertSliceLowering, LinSpaceLowering,
                LoadOpLowering, CreateLowering, EWBinOpLowering,
                EWUnyOpLowering, ReductionOpLowering, ReshapeLowering>(
            typeConverter, &ctxt);

    // populate function boundaries using our special type converter
    ::mlir::populateFunctionOpInterfaceTypeConversionPattern<
        ::mlir::func::FuncOp>(patterns, typeConverter4Func);
    ::mlir::populateReturnOpTypeConversionPattern(patterns, typeConverter4Func);
    ::mlir::populateCallOpTypeConversionPattern(patterns, typeConverter4Func);

    mlir::scf::populateSCFStructuralTypeConversionsAndLegality(
        typeConverter, patterns, target);

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
