//===- SubviewOp.cpp - PTensor dialect  --------------------------*- C++
//-*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the SubviewOp of the PTensor dialect.
/// Copied from NTensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

imex::ptensor::PTensorType imex::ptensor::SubviewOp::inferResultType(
    imex::ptensor::PTensorType sourceType,
    mlir::ArrayRef<int64_t> staticOffsets, mlir::ArrayRef<int64_t> staticSizes,
    mlir::ArrayRef<int64_t> staticStrides) {
  unsigned rank = sourceType.getRank();
  (void)rank;
  assert(staticOffsets.size() == rank && "staticOffsets length mismatch");
  assert(staticSizes.size() == rank && "staticSizes length mismatch");
  assert(staticStrides.size() == rank && "staticStrides length mismatch");
  return imex::ptensor::PTensorType::get(
      staticSizes, sourceType.getElementType(), sourceType.getEnvironment(),
      sourceType.getLayout());
}

imex::ptensor::PTensorType imex::ptensor::SubviewOp::inferResultType(
    imex::ptensor::PTensorType sourceShapedTensorType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferResultType(sourceShapedTensorType, staticOffsets,
                                    staticSizes, staticStrides);
}

imex::ptensor::PTensorType imex::ptensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ptensor::PTensorType sourceType,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides) {
  auto inferredType = inferResultType(sourceType, offsets, sizes, strides);
  assert(inferredType.getRank() >= static_cast<int64_t>(resultShape.size()) &&
         "expected ");
  if (inferredType.getRank() == static_cast<int64_t>(resultShape.size()))
    return inferredType;

  assert(mlir::computeRankReductionMask(inferredType.getShape(), resultShape)
             .has_value() &&
         "invalid rank reduction");

  return imex::ptensor::PTensorType::get(
      resultShape, sourceType.getElementType(), sourceType.getEnvironment(),
      sourceType.getLayout());
}

imex::ptensor::PTensorType imex::ptensor::SubviewOp::inferRankReducedResultType(
    mlir::ArrayRef<int64_t> resultShape, imex::ptensor::PTensorType sourceType,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  return SubviewOp::inferRankReducedResultType(
      resultShape, sourceType, staticOffsets, staticSizes, staticStrides);
}

// Build a SubViewOp with mixed static and dynamic entries and custom result
// type. If the type passed is nullptr, it is inferred.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ptensor::PTensorType resultType, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<int64_t> staticOffsets, staticSizes, staticStrides;
  mlir::SmallVector<mlir::Value> dynamicOffsets, dynamicSizes, dynamicStrides;
  dispatchIndexOpFoldResults(offsets, dynamicOffsets, staticOffsets);
  dispatchIndexOpFoldResults(sizes, dynamicSizes, staticSizes);
  dispatchIndexOpFoldResults(strides, dynamicStrides, staticStrides);
  auto sourceType = source.getType().cast<imex::ptensor::PTensorType>();
  // Structuring implementation this way avoids duplication between builders.
  if (!resultType) {
    resultType = imex::ptensor::SubviewOp::inferResultType(
        sourceType, staticOffsets, staticSizes, staticStrides);
  }
  build(b, result, resultType, source, dynamicOffsets, dynamicSizes,
        dynamicStrides, b.getDenseI64ArrayAttr(staticOffsets),
        b.getDenseI64ArrayAttr(staticSizes),
        b.getDenseI64ArrayAttr(staticStrides));
  result.addAttributes(attrs);
}

// Build a SubViewOp with mixed static and dynamic entries and inferred result
// type.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<mlir::OpFoldResult> offsets,
    mlir::ArrayRef<mlir::OpFoldResult> sizes,
    mlir::ArrayRef<mlir::OpFoldResult> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, imex::ptensor::PTensorType(), source, offsets, sizes,
        strides, attrs);
}

// Build a SubViewOp with static entries and inferred result type.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, source, offsetValues, sizeValues, strideValues, attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the
// type passed is nullptr, it is inferred.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ptensor::PTensorType resultType, mlir::Value source,
    mlir::ArrayRef<int64_t> offsets, mlir::ArrayRef<int64_t> sizes,
    mlir::ArrayRef<int64_t> strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues = llvm::to_vector<4>(
      llvm::map_range(offsets, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues = llvm::to_vector<4>(
      llvm::map_range(sizes, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues = llvm::to_vector<4>(
      llvm::map_range(strides, [&](int64_t v) -> mlir::OpFoldResult {
        return b.getI64IntegerAttr(v);
      }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues,
        attrs);
}

// Build a SubViewOp with dynamic entries and custom result type. If the type
// passed is nullptr, it is inferred.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result,
    imex::ptensor::PTensorType resultType, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  mlir::SmallVector<mlir::OpFoldResult> offsetValues =
      llvm::to_vector<4>(llvm::map_range(
          offsets, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> sizeValues =
      llvm::to_vector<4>(llvm::map_range(
          sizes, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  mlir::SmallVector<mlir::OpFoldResult> strideValues =
      llvm::to_vector<4>(llvm::map_range(
          strides, [](mlir::Value v) -> mlir::OpFoldResult { return v; }));
  build(b, result, resultType, source, offsetValues, sizeValues, strideValues);
}

// Build a SubViewOp with dynamic entries and inferred result type.
void imex::ptensor::SubviewOp::build(
    mlir::OpBuilder &b, mlir::OperationState &result, mlir::Value source,
    mlir::ValueRange offsets, mlir::ValueRange sizes, mlir::ValueRange strides,
    mlir::ArrayRef<mlir::NamedAttribute> attrs) {
  build(b, result, imex::ptensor::PTensorType(), source, offsets, sizes,
        strides, attrs);
}

// Copypasted from upstream tensor.
llvm::SmallBitVector imex::ptensor::SubviewOp::getDroppedDims() {
  mlir::ArrayRef<int64_t> resultShape = getType().getShape();
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims(mixedSizes.size());
  unsigned shapePos = 0;
  for (const auto &size : enumerate(mixedSizes)) {
    llvm::Optional<int64_t> sizeVal = getConstantIntValue(size.value());
    // If the size is not 1, or if the current matched dimension of the result
    // is the same static shape as the size value (which is 1), then the
    // dimension is preserved.
    if (!sizeVal || *sizeVal != 1 ||
        (shapePos < resultShape.size() && resultShape[shapePos] == 1)) {
      shapePos++;
      continue;
    }
    droppedDims.set(size.index());
  }
  return droppedDims;
}

// static bool isIdentitySubview(imex::ptensor::SubviewOp op) {
//   auto srcType = op.getSource().getType().cast<imex::ptensor::PTensorType>();
//   if (srcType != op.getResult().getType())
//     return false;

//   for (auto val : op.getMixedOffsets())
//     if (!mlir::isConstantIntValue(val, 0))
//       return false;

//   auto srcShape = srcType.getShape();
//   for (auto [i, val] : llvm::enumerate(op.getMixedSizes())) {
//     assert(i < srcShape.size());
//     auto shapeVal = srcShape[i];
//     if (mlir::ShapedType::isDynamic(shapeVal)) {
//       auto dim = val.dyn_cast<mlir::Value>();
//       if (!dim)
//         return false;

//       auto dimOp = dim.getDefiningOp<imex::ptensor::DimOp>();
//       if (!dimOp)
//         return false;

//       auto dimInd = dimOp.getConstantIndex();
//       if (!dimInd || *dimInd != static_cast<int64_t>(i))
//         return false;
//     } else {
//       if (!mlir::isConstantIntValue(val, shapeVal))
//         return false;
//     }
//   }

//   for (auto val : op.getMixedStrides())
//     if (!mlir::isConstantIntValue(val, 1))
//       return false;

//   return true;
// }

// mlir::OpFoldResult
// imex::ptensor::SubviewOp::fold(llvm::ArrayRef<mlir::Attribute> /*operands*/)
// {
//   if (isIdentitySubview(*this))
//     return getSource();

//   return nullptr;
// }

// Copypasted from upstream tensor.
mlir::LogicalResult imex::ptensor::SubviewOp::reifyResultShapes(
    mlir::OpBuilder &builder,
    mlir::ReifiedRankedShapedTypeDims &reifiedReturnShapes) {
  reifiedReturnShapes.resize(1);
  reifiedReturnShapes[0].reserve(getType().getRank());
  mlir::SmallVector<mlir::OpFoldResult> mixedSizes = getMixedSizes();
  llvm::SmallBitVector droppedDims = getDroppedDims();
  mlir::Location loc = getLoc();
  for (const auto &size : enumerate(mixedSizes)) {
    if (droppedDims.test(size.index()))
      continue;
    if (auto attr = size.value().dyn_cast<mlir::Attribute>()) {
      reifiedReturnShapes[0].push_back(
          builder.create<mlir::arith::ConstantIndexOp>(
              loc, attr.cast<mlir::IntegerAttr>().getInt()));
      continue;
    }
    reifiedReturnShapes[0].push_back(size.value().get<mlir::Value>());
  }
  return mlir::success();
}
