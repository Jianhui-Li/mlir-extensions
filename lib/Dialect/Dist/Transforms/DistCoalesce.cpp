//===- DistCoalesce.cpp - PTensorToDist Transform  -----*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms of the Dist dialect.
///
/// This pass tries to minimize the number of dist::RePartitionOps.
/// Instead of creating a new copy for each repartition, it tries to combine
/// multiple RePartitionOps into one. For this, it computes the local bounding
/// box of several uses of repartitioned copies of the same base tensor. It
/// replaces all matched RepartitionOps with one which provides the computed
/// bounding box. Uses of the eliminated RePartitionOps get updated with th
/// appropriate target part as originally used. Right now supported uses are
/// SubviewOps and InsertSliceOps.
///
/// InsertSliceOps are special because they mutate data. Hence they serve as
/// barriers across which no combination of RePartitionOps will happen.
///
/// Additionally, while most other ops do not request a special target part,
/// InsertSliceOps request a target part on the incoming tensor. This target
/// part gets back-propagated as far as possible, most importantly including
/// EWBinOps.
///
/// Also, as part of this back-propagation, RePartitionOps between two EWBinOps,
/// e.g. those which come from one EWBinOp and have only one use and that in a
/// another EWBinOp get simply erased.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>

#include "PassDetail.h"

#include <iostream>
#include <set>
#include <unordered_map>

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistCoalescePass : public ::imex::DistCoalesceBase<DistCoalescePass> {

  DistCoalescePass() = default;

  // returns true if a Value is defined by any of the given operation types
  template <typename T, typename... Ts>
  static bool isDefByAnyOf(const ::mlir::Value &val) {
    if (val.getDefiningOp<T>())
      return true;
    if constexpr (sizeof...(Ts))
      return isDefByAnyOf<Ts...>(val);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  // returns true if an operation is of any of the given types
  template <typename T, typename... Ts>
  static bool isAnyOf(const ::mlir::Operation *op) {
    if (::mlir::dyn_cast<T>(op))
      return true;
    if constexpr (sizeof...(Ts))
      return isAnyOf<Ts...>(op);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  /// Follow def-chain of given Value until hitting a creation function
  /// or EWBinOp
  /// @return defining op
  ::mlir::Operation *getBase(const ::mlir::Value &val) {
    if (auto op = val.getDefiningOp<::imex::dist::InitDistTensorOp>()) {
      auto pt = op.getPTensor();
      if (isDefByAnyOf<::imex::ptensor::CreateOp, ::imex::ptensor::ARangeOp,
                       ::imex::ptensor::EWBinOp, ::imex::ptensor::ReductionOp>(
              pt)) {
        return op;
      }
      return getBase(pt);
    } else if (auto op = val.getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return op;
    } else if (auto op = val.getDefiningOp<::imex::dist::SubviewOp>()) {
      return getBase(op.getSource());
    } else if (auto op = val.getDefiningOp<::imex::dist::InsertSliceOp>()) {
      return getBase(op.getDestination());
    } else {
      std::cerr << "oops. Unexpected op found: ";
      const_cast<::mlir::Value &>(val).dump();
      assert(false);
    }
  }

  /// return true if given op comes from a EWBinOp and has another EWBinOP
  /// as its single user.
  bool is_temp(::imex::dist::RePartitionOp &op) {
    if (op.getTargetSizes().size() == 0 && op->hasOneUse() &&
        ::mlir::isa<::imex::ptensor::EWBinOp>(*op->user_begin()) &&
        op.getBase().template getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return true;
    }
    return false;
  }

  /// update a SubviewOp with a target part
  /// create and return a new op if the SubviewOp has more than one use.
  ::mlir::Operation *updateTargetPart(::mlir::IRRewriter &builder,
                                      ::imex::dist::SubviewOp op,
                                      const ::mlir::ValueRange &tOffs,
                                      const ::mlir::ValueRange &tSizes) {

    // check if an existing target is the same as ours
    auto offs = op.getTargetOffsets();
    auto szs = op.getTargetSizes();
    if (offs.size() > 0) {
      assert(offs.size() == szs.size());
      for (size_t i = 0; i < offs.size(); ++i) {
        if ((tOffs[i] != offs[i] || tSizes[i] != szs[i]) && !op->hasOneUse()) {
          // existing but different target -> need a new repartition for our
          // back-propagation
          auto val = op.getSource();
          builder.setInsertionPointAfter(op);
          return builder.create<::imex::dist::RePartitionOp>(
              op->getLoc(), val.getType(), val, tOffs, tSizes);
        }
      }
      // if same existing target -> nothing to be done
    } else {
      // no existing target -> use ours
      op->insertOperands(op->getNumOperands(), tOffs);
      auto nOpsBefore = op->getNumOperands();
      op->insertOperands(op->getNumOperands(), tSizes);

      // target sizes might be larger than our static sizes
      // FIXME dynamic broadcasting needs to check dyn sizes, too
      const int32_t rank = static_cast<int32_t>(tOffs.size());
      for (int32_t i = 0; i < rank; ++i) {
        auto s = op.getStaticSizes()[i];
        if (!::mlir::ShapedType::isDynamic(s)) {
          op->setOperand(i + nOpsBefore, createIndex(op->getLoc(), builder, s));
          // if the static size is 1, then we need to duplicate
          // -> target offset is the same on all procs: 0
          if (s == 1) {
            op->setOperand(i + nOpsBefore - rank,
                           createIndex(op->getLoc(), builder, 0));
          }
        }
      }

      const auto sSzsName = op.getOperandSegmentSizesAttrName();
      const auto oa = op->getAttrOfType<::mlir::DenseI32ArrayAttr>(sSzsName);
      ::std::array<int32_t, 6> sSzs{oa[0], oa[1], oa[2], oa[3], rank, rank};
      op->setAttr(sSzsName, builder.getDenseI32ArrayAttr(sSzs));
    }
    return nullptr;
  }

  /// @return if op1 can be moved directly after op2
  bool canMoveAfter(::mlir::DominanceInfo &dom, ::mlir::Operation *op1,
                    ::mlir::Operation *op2) {
    assert(op2 != op1);
    if (dom.dominates(op2, op1)) {
      for (auto o : op1->getOperands()) {
        auto dOp = o.getDefiningOp();
        if (dOp && !dom.dominates(dOp, op2)) {
          return false;
        }
      }
    }
    return true;
  }

  /// entry point for back propagation of target parts, starting with
  /// RePartitionOp. Verifies that defining ops are what we assume/can handle.
  /// Then starts actual back propagation
  uint64_t
  backPropagatePart(::mlir::IRRewriter &builder, ::mlir::DominanceInfo &dom,
                    ::imex::dist::RePartitionOp rpOp, ::mlir::Operation *&nOp,
                    ::std::set<::imex::dist::RePartitionOp> &toDelete) {
    nOp = nullptr;
    auto offs = rpOp.getTargetOffsets();
    if (offs.empty()) {
      return 0;
    }

    auto defOp2 = offs[0].getDefiningOp();
    if (defOp2) {
      auto ltosOp = mlir::dyn_cast<::imex::dist::LocalTargetOfSliceOp>(defOp2);
      assert(ltosOp);
      auto fOp = ltosOp.getDTensor().getDefiningOp();
      assert(fOp);
      if (canMoveAfter(dom, defOp2, fOp)) {
        defOp2->moveAfter(fOp);
      } else {
        // this is pretty strict, we might miss potential
        return 0;
      }
    }
    // else would mean it's a block arg which is fine anyway

    auto szs = rpOp.getTargetSizes();
    return backPropagatePart(builder, rpOp, offs, szs, nOp, toDelete);
  }

  /// The actual back propagation of target parts
  /// if meeting a supported op, recursively gets defining ops and back
  /// propagates as it follows only supported ops, all other ops act as
  /// propagation barriers (e.g. InsertSliceOps) on the way it updates target
  /// info on SubviewOps and marks RePartitionOps for elimination
  uint64_t
  backPropagatePart(::mlir::IRRewriter &builder, ::mlir::Operation *op,
                    const ::mlir::ValueRange &tOffs,
                    const ::mlir::ValueRange &tSizes, ::mlir::Operation *&nOp,
                    ::std::set<::imex::dist::RePartitionOp> &toDelete) {
    ::mlir::Value val;
    uint64_t n = 0;
    nOp = nullptr;
    if (auto typedOp = ::mlir::dyn_cast<::imex::dist::SubviewOp>(op)) {
      val = typedOp.getSource();
      updateTargetPart(builder, typedOp, tOffs, tSizes);
    } else if (auto typedOp =
                   ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
      // continue even if already deleted in case different target parts are
      // needed
      val = typedOp.getBase();
      toDelete.emplace(typedOp);
    } else if (auto typedOp = ::mlir::dyn_cast<::imex::ptensor::EWBinOp>(op)) {
      auto defOp = typedOp.getLhs().getDefiningOp();
      if (defOp) {
        n = backPropagatePart(builder, defOp, tOffs, tSizes, nOp, toDelete);
        assert(!nOp || !"not implemented yet");
      }
      val = typedOp.getRhs();
    }
    ::mlir::Operation *defOp = nullptr;
    if (val) {
      defOp = val.getDefiningOp();
      ++n;
    }
    return defOp ? n + backPropagatePart(builder, defOp, tOffs, tSizes, nOp,
                                         toDelete)
                 : n;
  }

  // This pass tries to combine multiple RePartitionOps into one.
  // Dependent operations (like SubviewOp) get adequately annotated.
  //
  // The basic idea is to compute a the bounding box of several RePartitionOps
  // and use it for a single repartition. Dependent SubviewOps can then
  // extract the appropriate part from that bounding box without further
  // communication/repartitioning.
  //
  // 1. back-propagation of explicit target-parts
  // 2. group and move SubviewOps
  // 3. create base RePartitionOps and update dependent SubviewOps
  void runOnOperation() override {

    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());

    // back-propagate targets from RePartitionOps

    ::std::set<::imex::dist::RePartitionOp> rpToElimNew;
    ::mlir::SmallVector<::imex::dist::RePartitionOp> rpOps;

    // store all RePartitionOps with target in vector
    root->walk([&](::mlir::Operation *op) {
      builder.setInsertionPoint(op);
      return ::mlir::WalkResult::interrupt();
    });

    // store all RePartitionOps with target in vector
    root->walk([&](::imex::dist::RePartitionOp op) {
      if (!op.getTargetOffsets().empty()) {
        rpOps.emplace_back(op);
      }
    });

    auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

    // perform back propagation on each
    for (auto rp = rpOps.rbegin(); rp != rpOps.rend(); ++rp) {
      if (rpToElimNew.find(*rp) == rpToElimNew.end()) {
        ::mlir::Operation *nOp = nullptr;
        backPropagatePart(builder, dom, *rp, nOp, rpToElimNew);
        assert(!nOp);
      }
    }

    // eliminate no longer needed RePartitionOps
    for (auto rp : rpToElimNew) {
      builder.replaceOp(rp, rp.getBase());
    }

    // find InsertSliceOp, SubviewOp and RePartitionOps on the same base
    // pointer
    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer

    std::unordered_map<::mlir::Operation *,
                       ::mlir::SmallVector<::mlir::Operation *>>
        opsGroups;
    root->walk([&](::mlir::Operation *op) {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp = ::mlir::dyn_cast<::imex::dist::SubviewOp>(op)) {
        val = typedOp.getSource();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
        val = typedOp.getBase();
      }
      if (val) {
        auto base = getBase(val);
        opsGroups[base].emplace_back(op);
      }
    });

    // outer loop iterates base over base pointers
    for (auto grpP : opsGroups) {
      if (grpP.second.size() < 1)
        continue;

      auto &base = grpP.first;
      auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

      builder.setInsertionPointAfter(base);
      auto team = createTeamOf(base->getLoc(), builder, base->getResult(0));
      auto nProcs = createNProcs(base->getLoc(), builder, team);
      auto pRank = createPRank(base->getLoc(), builder, team);

      // find groups operating on the same base, groups are separated by write
      // operations (InsertSliceOps for now)
      for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j) {
        ::mlir::SmallVector<::mlir::Operation *> grp;
        ::mlir::SmallVector<::mlir::Operation *> unhandled;
        int nEx = 0;

        for (auto i = j; i != grpP.second.end(); ++i, ++j) {
          if (::mlir::dyn_cast<::imex::dist::InsertSliceOp>(*i)) {
            break;
          }
          grp.emplace_back(*i);
          if (::mlir::dyn_cast<::imex::dist::SubviewOp>(*i)) {
            ++nEx;
          }
        }

        // iterate over group until all ops are handled
        // we might not be able to move all SubviewOps to the point which
        // is early enough to have a single repartition. Hence we have to loop
        // until we handled all sub-groups.
        while (grp.size() > 0) {
          ::mlir::SmallVector<::imex::dist::RePartitionOp> rpToElim;
          auto fOp = grp.front();
          ::mlir::Operation *eIPnt = nullptr;
          auto rpIPnt = fOp;
          auto bbIPnt = fOp;
          ::mlir::ValueRange bbOffs, bbSizes;
          ::mlir::Operation *combined = nullptr;

          // iterate group
          for (auto i = grp.begin(); i != grp.end(); ++i) {
            auto e = ::mlir::dyn_cast<::imex::dist::SubviewOp>(*i);
            auto rp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(*i);
            // check if we can move current op up
            if (dom.dominates(fOp, *i)) {
              bool can_move = true;
              if (fOp != *i) {
                for (auto o : (*i)->getOperands()) {
                  if (!dom.dominates(o.getDefiningOp(), e ? eIPnt : rpIPnt)) {
                    can_move = false;
                    break;
                  }
                }
              }

              // if it's safe to move: do it
              if (can_move) {
                if (e) {
                  if (false && nEx < 2) {
                    e->moveBefore(rpIPnt);
                  } else {
                    auto loc = e.getLoc();
                    builder.setInsertionPointAfter(bbIPnt);
                    ::mlir::ValueRange tOffs = e.getTargetOffsets();
                    ::mlir::ValueRange tSizes = e.getTargetSizes();
                    if (tOffs.empty()) {
                      assert(tSizes.empty());
                      auto lPart =
                          builder.create<::imex::dist::LocalPartitionOp>(
                              loc, nProcs, pRank, e.getSizes());
                      tOffs = lPart.getLOffsets();
                      tSizes = lPart.getLShape();
                      bbIPnt = lPart;
                      updateTargetPart(builder, e, tOffs, tSizes);
                    }

                    auto rank = tOffs.size();
                    // extend local bounding box
                    auto _offs = getMixedAsValues(loc, builder, e.getOffsets(),
                                                  e.getStaticOffsets());
                    auto _sizes = getMixedAsValues(loc, builder, e.getSizes(),
                                                   e.getStaticSizes());
                    auto _strides = getMixedAsValues(
                        loc, builder, e.getStrides(), e.getStaticStrides());

                    auto bbox =
                        builder.create<::imex::dist::LocalBoundingBoxOp>(
                            loc, base->getResult(0), _offs, _sizes, _strides,
                            tOffs, tSizes, bbOffs, bbSizes);
                    bbOffs = bbox.getResultOffsets();
                    bbSizes = bbox.getResultSizes();
                    bbIPnt = bbox;
                    if (combined) {
                      combined->setOperands(1, rank, bbOffs);
                      combined->setOperands(1 + rank, rank, bbSizes);
                    } else {
                      combined = builder.create<::imex::dist::RePartitionOp>(
                          loc, base->getResult(0).getType(), base->getResult(0),
                          bbOffs, bbSizes);
                    }
                    e->moveAfter(eIPnt ? eIPnt : combined);
                    e->setOperand(0, combined->getResult(0));
                    eIPnt = *i;
                    // any repartitionops of this extract slice can potentially
                    // be eliminated
                    for (auto u : e->getUsers()) {
                      if (auto r =
                              ::mlir::dyn_cast<::imex::dist::RePartitionOp>(
                                  u)) {
                        rpToElim.emplace_back(u);
                      }
                    }
                  }
                } else {
                  assert(rp);
                  (*i)->moveAfter(rpIPnt);
                  rpIPnt = *i;
                }
                continue;
              }
            }
            // if fOp does not dominate i or i's inputs do not dominate fOp
            // we try later with remaining unhandled ops
            unhandled.emplace_back(*i);
          }

          // FIXME: handling of remaining repartitionops needs simplification
          for (auto o : rpToElim) {
            for (auto x : grp) {
              // elmiminate only if it is in our current group
              if (x == o) {
                assert(o.getTargetOffsets().empty());
                // remove from unhandled
                for (auto it = unhandled.begin(); it != unhandled.end(); ++it) {
                  if (*it == o) {
                    unhandled.erase(it);
                    break;
                  }
                }
                builder.replaceOp(o, o.getBase());
                break;
              }
            }
          }
          grp.clear();
          grp.swap(unhandled);
        }
        if (j == grpP.second.end())
          break;
      }
    }
  }
};

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
