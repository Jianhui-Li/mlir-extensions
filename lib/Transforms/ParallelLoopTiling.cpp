//===- ParallelLoopTiling.cpp - Tiles scf.parallel ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop tiling on parallel loops.
//
//===----------------------------------------------------------------------===//

#include "imex/Transforms/Passes.h"

#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/SCF/Transforms/Transforms.h>
#include <mlir/Dialect/SCF/Utils/Utils.h>

namespace imex {
#define GEN_PASS_DEF_LOOPTILINGONSCFPARALLEL
#include "imex/Transforms/Passes.h.inc"
} // namespace imex

using namespace imex;
using namespace mlir;
using namespace mlir::scf;

namespace mlir {
namespace scf {
std::pair<ParallelOp, ParallelOp> tileParallelLoop(ParallelOp op,
                                                   ArrayRef<int64_t> tileSizes,
                                                   bool noMinMaxBounds);
}
} // namespace mlir

// Compute the tile size based on the architecture info and loop properties.
// FIXME: better compute logic
static bool computeAndSetTileSize(ParallelOp op,
                                  SmallVector<int64_t> &tileSizes) {
  auto numLoops = op.getNumLoops();

  if (numLoops > 1)
    return false;
  auto upperBound = op.getUpperBound().front();
  int64_t value;
  if (auto IndexOp = upperBound.getDefiningOp<arith::ConstantIndexOp>()) {
    value = IndexOp.value();
  } else if (auto IntOp = upperBound.getDefiningOp<arith::ConstantIntOp>()) {
    value = IntOp.value();
  } else {
    return false;
  }
  auto tile = value < 64 ? value : 128;
  tileSizes.push_back(tile);
  return true;
}

namespace {
class LoopTilingOnSCFParallel
    : public imex::impl::LoopTilingOnSCFParallelBase<LoopTilingOnSCFParallel> {
public:
  void runOnOperation() override {
    auto *parentOp = getOperation();
    SmallVector<ParallelOp, 2> innermostPloops;
    getInnermostParallelLoops(parentOp, innermostPloops);
    for (ParallelOp ploop : innermostPloops) {
      // FIXME: Add reduction support.
      SmallVector<int64_t> tileSizes;
      auto do_tile = computeAndSetTileSize(ploop, tileSizes);
      if (do_tile && ploop.getNumReductions() == 0)
        tileParallelLoop(ploop, tileSizes, false);
    }
  }
};
} // namespace

namespace imex {
std::unique_ptr<mlir::Pass> createParallelLoopTilingPass() {
  return std::make_unique<LoopTilingOnSCFParallel>();
}
} // namespace imex
