//===- DimOp.cpp - PTensor dialect  --------------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the DimOp of the PTensor dialect.
/// Copied from pTensor.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/DialectImplementation.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/OpImplementation.h>

void imex::ptensor::DimOp::getAsmResultNames(
    llvm::function_ref<void(mlir::Value, mlir::StringRef)> setNameFn) {
  setNameFn(getResult(), "dim");
}

void imex::ptensor::DimOp::build(mlir::OpBuilder &builder,
                                 mlir::OperationState &result,
                                 mlir::Value source, int64_t index) {
  auto loc = result.location;
  auto indexValue = builder.create<mlir::arith::ConstantIndexOp>(loc, index);
  build(builder, result, source, indexValue);
}

llvm::Optional<int64_t> imex::ptensor::DimOp::getConstantIndex() {
  if (auto val = mlir::getConstantIntValue(getIndex()))
    return *val;

  return {};
}

mlir::Speculation::Speculatability imex::ptensor::DimOp::getSpeculatability() {
  auto constantIndex = getConstantIndex();
  if (!constantIndex)
    return mlir::Speculation::NotSpeculatable;

  auto rankedType =
      mlir::dyn_cast<imex::ptensor::PTensorType>(getSource().getType());
  if (!rankedType)
    return mlir::Speculation::NotSpeculatable;

  // The verifier rejects operations that violate this assertion.
  assert(constantIndex < rankedType.getRank());
  return mlir::Speculation::Speculatable;
}

mlir::LogicalResult imex::ptensor::DimOp::verify() {
  // Assume unknown index to be in range.
  llvm::Optional<int64_t> index = getConstantIndex();
  if (!index)
    return mlir::success();

  // Check that constant index is not knowingly out of range.
  auto type = getSource().getType();
  if (auto tensorType = type.dyn_cast<imex::ptensor::PTensorType>()) {
    if (*index >= tensorType.getRank())
      return emitOpError("index is out of range");
  } else {
    llvm_unreachable("expected operand with array type");
  }
  return mlir::success();
}

namespace {
// TODO: upstream
struct LinalgGenericDimPropagate
    : public mlir::OpRewritePattern<mlir::tensor::DimOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::DimOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto src = op.getSource();
    auto generic = src.getDefiningOp<mlir::linalg::GenericOp>();
    if (!generic)
      return mlir::failure();

    assert(generic.getOutputs().size() == generic.getResults().size());
    auto outIndex = [&]() -> size_t {
      for (auto [i, out] : llvm::enumerate(generic.getResults())) {
        if (out == src)
          return i;
      }
      llvm_unreachable("Invalid result");
    }();

    auto out = generic.getOutputs()[outIndex];

    rewriter.replaceOpWithNewOp<mlir::tensor::DimOp>(op, out, op.getIndex());
    return mlir::success();
  }
};
} // namespace

void imex::ptensor::DimOp::getCanonicalizationPatterns(
    ::mlir::RewritePatternSet &results, ::mlir::MLIRContext *context) {
  results.insert<LinalgGenericDimPropagate>(context);
}
