//===- CollapseDimensions.cpp - LoopCollapsePass  -------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file converts multi-diemntional element-wise linalg operator into
/// 1-dimentional linalg ops. It helps for achieving better performance by
/// enabling a good gpu mapping
//===----------------------------------------------------------------------===//

#include <imex/Transforms/Passes.h>

#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/GPU/Transforms/Passes.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/Linalg/IR/LinalgInterfaces.h>
#include <mlir/Dialect/Linalg/Transforms/Transforms.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Block.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/FunctionInterfaces.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LogicalResult.h>

#include <mlir/Pass/Pass.h>

using namespace mlir;
using namespace imex;

namespace imex {
#define GEN_PASS_DEF_COLLAPSEDIMENSIONS
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

/// Returns true if the given op is collapsable.
static bool isEligibleForCollapse(linalg::GenericOp genericOp) {
  // TODO(guray) There is no mechanism to tell the collapsed indexes to
  // `tensor.expand_shape`. Once we have this support in MLIR, we can enable
  // dynamic tensor shapes.
  if (genericOp.hasDynamicShape())
    return false;

  // TODO(guray) Currently we can only collapse when result of all the
  // AffineMaps are dimensions. Possible to collapse cases like
  // affine_map<d-2, d1+d2> with affine_map<d0, d1+d2>, however, this is not
  // supported in collapsing mechanism in MLIR. Once we have this support,
  // we can remove this if statement.
  if (llvm::any_of(genericOp.getIndexingMapsArray(), [](AffineMap map) {
        return !map.isProjectedPermutation();
      })) {
    return false;
  }

  // TODO(guray) Collapsing caused performance regression in a cpu
  // benchmark, so we disable it.
  if (genericOp.hasIndexSemantics())
    return false;

  return true;
}

/// Searches the same sequence in all the affine maps and collapses these
/// dimensions. It only applies these to "parallel" loops without mixing them
/// with "reduction" types.
static SmallVector<ReassociationIndices>
getCollapsibleLoops(linalg::GenericOp genericOp) {
  SmallVector<ReassociationIndices> contiguousLoops;

  SmallVector<unsigned> pDims;
  genericOp.getParallelDims(pDims);
  if (pDims.size() < 2)
    return contiguousLoops;

  llvm::SmallDenseSet<unsigned> pLoops(pDims.begin(), pDims.end());

  auto hasAllMapsSameSequence = [&](AffineExpr preExpr, AffineExpr nextExpr) {
    for (AffineMap map : genericOp.getIndexingMapsArray()) {
      bool foundSeq = false;
      for (auto [index, resultExpr] : llvm::enumerate(map.getResults())) {
        if (resultExpr == nextExpr) {
          foundSeq = (index > 0 && preExpr == map.getResult(index - 1));
          break;
        }
      }
      if (!foundSeq)
        return false;
    }
    return true;
  };

  ReassociationIndices range;
  AffineExpr preExpr;
  for (auto nextExpr : genericOp.getIndexingMapsArray().front().getResults()) {
    unsigned pos = nextExpr.cast<AffineDimExpr>().getPosition();
    if (!range.empty()) {
      if (!hasAllMapsSameSequence(preExpr, nextExpr) || !pLoops.count(pos)) {
        if (range.size() > 1)
          contiguousLoops.push_back({range.begin(), range.end()});
        range.clear();
      }
    }
    preExpr = nextExpr;
    if (pLoops.count(pos))
      range.push_back(pos);
  }
  if (range.size() > 1)
    contiguousLoops.push_back(range);

  return contiguousLoops;
}

/// Collapse possible dimension of the given linalg.generic
static FailureOr<SmallVector<Value>>
collapseLinalgGeneric(IRRewriter &rewriter, linalg::GenericOp genericOp,
                      SmallVector<ReassociationIndices> &collapseIndices) {
  rewriter.setInsertionPoint(genericOp);
  FailureOr<SmallVector<Value>> replacements =
      mlir::linalg::collapseGenericOpIterationDims(genericOp, collapseIndices,
                                                   rewriter);
  if (failed(replacements) || replacements->empty()) {
    return rewriter.notifyMatchFailure(genericOp,
                                       "failed to collapse dimensions");
  }

  return replacements;
}

/// Traverses DispatchRegionOps to find linalg genericOps that has no
/// producers and tries to collapse its dimensions.
static LogicalResult collapseDimensions(IRRewriter &rewriter,
                                        linalg::GenericOp &genericOp) {

  // Step 1. Check whether it is possible to collapse
  if (!isEligibleForCollapse(genericOp))
    return success();

  rewriter.setInsertionPoint(genericOp);

  SmallVector<ReassociationIndices> collapseIndices;
  collapseIndices = getCollapsibleLoops(genericOp);
  if (collapseIndices.empty())
    return success();

  // Step 2. Collapse dimensions
  auto maybeReplacements =
      collapseLinalgGeneric(rewriter, genericOp, collapseIndices);
  if (failed(maybeReplacements)) {
    return failure();
  }

  rewriter.replaceOp(genericOp, *maybeReplacements);

  return success();
}

namespace {
class CollapseDimensionsPass final
    : public imex::impl::CollapseDimensionsBase<CollapseDimensionsPass> {

public:
  void runOnOperation() override {
    auto funcOp = getOperation();
    IRRewriter rewriter(funcOp->getContext());
    funcOp->walk([&](linalg::GenericOp genericOp) {
      if (failed(collapseDimensions(rewriter, genericOp)))
        return signalPassFailure();
    });
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createCollapseDimensionsPass() {
  return std::make_unique<CollapseDimensionsPass>();
}
} // namespace imex
