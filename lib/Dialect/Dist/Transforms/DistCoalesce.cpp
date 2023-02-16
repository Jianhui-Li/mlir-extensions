//===- DistCoalesce.cpp - PTensorToDist Transform  -----*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements transforms of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/IR/DistOps.h>
#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>

#include <mlir/Analysis/AliasAnalysis/LocalAliasAnalysis.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>

#include "PassDetail.h"

#include <iostream>
#include <unordered_map>
#include <vector>

namespace imex {
namespace dist {

namespace {

// *******************************
// ***** Some helper functions ***
// *******************************

// Return number of ranks/processes in given team/communicator
// uint64_t idtr_nprocs(int64_t team);

// *******************************
// ***** Pass infrastructure *****
// *******************************

// Lowering dist dialect by no-ops
struct DistCoalescePass : public ::imex::DistCoalesceBase<DistCoalescePass> {

  DistCoalescePass() = default;

  template <typename T, typename... Ts>
  static bool isDefByAnyOf(const ::mlir::Value &val) {
    if (val.getDefiningOp<T>())
      return true;
    if constexpr (sizeof...(Ts))
      return isDefByAnyOf<Ts...>(val);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  template <typename T, typename... Ts>
  static bool isAnyOf(const ::mlir::Operation *op) {
    if (::mlir::dyn_cast<T>(op))
      return true;
    if constexpr (sizeof...(Ts))
      return isAnyOf<Ts...>(op);
    else if constexpr (!sizeof...(Ts))
      return false;
  }

  using rb_info =
      ::std::pair<::mlir::Operation *,
                  ::mlir::SmallVector<::imex::dist::ExtractSliceOp>>;

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
    } else if (auto op = val.getDefiningOp<::imex::dist::ExtractSliceOp>()) {
      return getBase(op.getSource());
    } else if (auto op = val.getDefiningOp<::imex::dist::InsertSliceOp>()) {
      return getBase(op.getDestination());
    } else {
      std::cerr << "oops. Unexpected op found: ";
      const_cast<::mlir::Value &>(val).dump();
      assert(false);
    }
  }

  bool is_temp(::imex::dist::RePartitionOp &op) {
    if (op.getTargetSizes().size() == 0 && op->hasOneUse() &&
        ::mlir::isa<::imex::ptensor::EWBinOp>(*op->user_begin()) &&
        op.getBase().template getDefiningOp<::imex::ptensor::EWBinOp>()) {
      return true;
    }
    return false;
  }

#if 0
    // singularize calls to nprocs and prank
    auto singularize = [](auto &op1, auto &op2) {
      return op1.getTeam() == op2.getTeam();
    };
    groupOps<::imex::dist::NProcsOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::imex::dist::NProcsOp &op) { return true; },
        [](::imex::dist::NProcsOp &op) { return std::array<::mlir::Value, 1>{op.getTeam()}; },
        singularize);
    groupOps<::imex::dist::PRankOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), getOperation(),
        [](::imex::dist::PRankOp &op) { return true; },
        [](::imex::dist::PRankOp &op) { return std::array<::mlir::Value, 1>{op.getTeam()}; },
        singularize);
#endif
#if 0
    // group all create ops
    groupOps<::imex::ptensor::CreateOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), root,
        [](::imex::ptensor::CreateOp &op) { return true; },
        [](::imex::ptensor::CreateOp &op) { return op.getOperands(); },
        [](::imex::ptensor::CreateOp &, ::imex::ptensor::CreateOp &) {
          return false;
        }, [](auto, auto, auto&){return false;});
    // group all repartition ops
    groupOps<::imex::dist::RePartitionOp>(
        this->getAnalysis<::mlir::DominanceInfo>(), root,
        [](::imex::dist::RePartitionOp &op) { return true; },
        [](::imex::dist::RePartitionOp &op) { return op.getOperands(); },
        [](::imex::dist::RePartitionOp &, ::imex::dist::RePartitionOp &) {
          return false;
        }, hasWriteBetween);

    using op_ex_pair =
        ::std::pair<::imex::dist::RePartitionOp,
                    ::mlir::SmallVector<::imex::dist::ExtractSliceOp>>;
    std::vector<std::unordered_map<::mlir::Operation *,
                                   ::mlir::SmallVector<op_ex_pair>>>
        all_rbs;
    std::vector<::imex::dist::RePartitionOp> tmpOps;
#endif

  void runOnOperation() override {

    auto root = this->getOperation();
    ::mlir::IRRewriter builder(&getContext());

    // Find all repartition operations which are temporaries in between EWBinOps
    {
      ::mlir::SmallVector<::imex::dist::RePartitionOp> tmpRPs;
      root->walk([&](::mlir::Operation *op) {
        if (auto typedOp = ::mlir::dyn_cast<::imex::dist::RePartitionOp>(op)) {
          if (is_temp(typedOp)) {
            tmpRPs.emplace_back(typedOp);
          }
        }
      });

      // we can simply eliminate such temporaries
      for (auto rp : tmpRPs) {
        builder.replaceOp(rp, rp.getBase());
      }
    }

    // find insertslice, extractslice and repartition ops on the same base
    // pointer
    std::unordered_map<::mlir::Operation *,
                       ::mlir::SmallVector<::mlir::Operation *>>
        opsGroups;
    root->walk([&](::mlir::Operation *op) {
      ::mlir::Value val;
      if (auto typedOp = ::mlir::dyn_cast<::imex::dist::InsertSliceOp>(op)) {
        val = typedOp.getDestination();
      } else if (auto typedOp =
                     ::mlir::dyn_cast<::imex::dist::ExtractSliceOp>(op)) {
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

    // opsGroups holds independent partial operation sequences operating on a
    // specific base pointer

    for (auto grpP : opsGroups) {
      if (grpP.second.size() <= 1)
        continue;
      // assert(::mlir::dyn_cast<::imex::dist::ExtractSliceOp>(grpP.second.front()));

      auto &base = grpP.first;
      auto &dom = this->getAnalysis<::mlir::DominanceInfo>();

      builder.setInsertionPointAfter(base);
      auto team = createTeamOf(base->getLoc(), builder, base->getResult(0));
      auto nProcs = createNProcs(base->getLoc(), builder, team);
      auto pRank = createPRank(base->getLoc(), builder, team);

      for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j) {
        std::cerr << "yyy: ";
        (*j)->dump();
        std::cerr << std::endl;
      }
      for (auto j = grpP.second.begin(); j != grpP.second.end(); ++j) {
        ::mlir::SmallVector<::mlir::Operation *> grp;
        ::mlir::SmallVector<::mlir::Operation *> unhandled;
        int nEx = 0;

        for (auto i = j; i != grpP.second.end(); ++i, ++j) {
          if (::mlir::dyn_cast<::imex::dist::InsertSliceOp>(*i)) {
            break;
          }
          std::cerr << "xxx: ";
          (*i)->dump();
          std::cerr << std::endl;
          grp.emplace_back(*i);
          if (::mlir::dyn_cast<::imex::dist::ExtractSliceOp>(*i)) {
            ++nEx;
          }
        }

        while (grp.size() > 1) {
          ::mlir::SmallVector<::imex::dist::RePartitionOp> rpToElim;
          auto fOp = grp.front();
          ::mlir::Operation *eIPnt = nullptr;
          auto rpIPnt = fOp;
          auto bbIPnt = fOp;
          ::mlir::ValueRange bbOffs, bbSizes;
          ::mlir::Operation *combined = nullptr;

          for (auto i = grp.begin(); i != grp.end(); ++i) {
            auto e = ::mlir::dyn_cast<::imex::dist::ExtractSliceOp>(*i);
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
                  if (nEx < 2) {
                    e->moveBefore(rpIPnt);
                  } else {
                    auto loc = e.getLoc();
                    builder.setInsertionPointAfter(bbIPnt);
                    auto lPart = builder.create<::imex::dist::LocalPartitionOp>(
                        loc, nProcs, pRank, e.getSizes());
                    ::mlir::ValueRange tOffs = lPart.getLOffsets();
                    ::mlir::ValueRange tSizes = lPart.getLShape();
                    bbIPnt = lPart;

                    auto rank = tOffs.size();
                    // extend local bounding box
                    auto bbox =
                        builder.create<::imex::dist::LocalBoundingBoxOp>(
                            loc, base->getResult(0), e.getOffsets(),
                            e.getSizes(), e.getStrides(), tOffs, tSizes, bbOffs,
                            bbSizes);
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
                    e->insertOperands(e->getNumOperands(), tOffs);
                    e->insertOperands(e->getNumOperands(), tSizes);
                    ::mlir::SmallVector<int32_t> sSzs(6, rank);
                    sSzs[0] = 1;
                    e->setAttr(e.getOperandSegmentSizesAttrName(),
                               builder.getDenseI32ArrayAttr(sSzs));
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
          for (auto o : rpToElim) {
            for (auto x : grp) {
              // elmiminate only if it is in our current group
              if (x == o) {
                assert(o.getTargetOffsets().empty());
                std::cerr << "rp: ";
                o.dump();
                std::cerr << std::endl;
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

#if 0

    for (auto &rbs : all_rbs) {
      // for each base tensor combine views on it
      for (auto &[base, ops] : rbs) {
        ::mlir::SmallVector<::mlir::Value> _views;
        ::mlir::SmallVector<::imex::dist::ExtractSliceOp> _extracts;
        ::mlir::SmallVector<::imex::dist::RePartitionOp> _rpops;

        // visit each view and store view info in _views and the extractslice op
        // in _extracts
        for (unsigned i = 0; i < ops.size(); ++i) {
          if (ops[i].second.size() > 0) {
            // _views.emplace_back(ops[i].first.getDTensor());
            _extracts.emplace_back(ops[i].second.front());
            _rpops.emplace_back(ops[i].first);
          } else
            assert(ops[i].second.empty());
          // FIXME no support of views of views yet
        }

        // nothing to be done if there are not multiple views
        // FIXME as long as our target partition is balanced we can just erase
        // the repartition
        if (_rpops.size() <= 1) {
          // for(auto o : ops) {
          //   // eliminate related repartition ops
          //   builder.replaceOp(o.first, o.first->getOperand(0));
          // }
          continue;
        }

        // ::mlir::ValueRange views{_views};

        // auto idxTyp = builder.getIndexType();
        // auto t1Typ = ::mlir::TupleType::get(builder.getContext(),
        // ::std::SmallVector<::mlir::Type>(rank, idxTyp)); auto t2Typ =
        // ::mlir::TupleType::get(builder.getContext(),
        // ::std::array<::mlir::Type>(3, t1Typ)); auto t3Typ =
        // ::mlir::TupleType::get(builder.getContext(),
        // ::std::SmallVector<::mlir::Type>(_extracts.size(), t2Typ));

        ::imex::dist::LocalBoundingBoxOp bbox(nullptr);
        ::mlir::SmallVector<::mlir::ValueRange> tOffsVec, tSizesVec;

        builder.setInsertionPoint(_rpops.front());
        auto team = createTeamOf(base->getLoc(), builder, base->getResult(0));
        auto nProcs = createNProcs(base->getLoc(), builder, team);
        auto pRank = createPRank(base->getLoc(), builder, team);

        for (unsigned i = 0; i < _extracts.size(); ++i) {
          auto e = _extracts[i];
          auto o = _rpops[i];
          auto loc = e.getLoc();

          // FIXME we use default partitioning here, we should back-propagate
          ::mlir::ValueRange tOffs = o.getTargetOffsets();
          ::mlir::ValueRange tSizes = o.getTargetSizes();
          if (tSizes.size() == 0) {
            auto lPart = builder.create<::imex::dist::LocalPartitionOp>(
                loc, nProcs, pRank, e.getSizes());
            tOffs = lPart.getLOffsets();
            tSizes = lPart.getLShape();
          }
          tOffsVec.emplace_back(tOffs);
          tSizesVec.emplace_back(tSizes);

          ::mlir::ValueRange offs, szs;
          if (i > 0) {
            if (i == 1) {
              offs = tOffsVec[0];
              szs = tSizesVec[0];
            } else {
              offs = bbox.getResultOffsets();
              szs = bbox.getResultSizes();
            }
            // extend local bounding box
            bbox = builder.create<::imex::dist::LocalBoundingBoxOp>(
                loc, base->getResult(0), e.getOffsets(), e.getSizes(),
                e.getStrides(), tOffs, tSizes, offs, szs);
          }
        }

        auto combined =
            createRePartition(base->getLoc(), builder, base->getResult(0),
                              bbox.getResultOffsets(), bbox.getResultSizes());

        // finally update all related extractslice ops
        for (unsigned i = 0; i < _extracts.size(); ++i) {
          auto e = _extracts[i];
          auto o = _rpops[i];
          auto tOffs = tOffsVec[i];
          auto tSizes = tSizesVec[i];
          // right now we support only balanced target partitions
          assert(e.getTargetOffsets().empty() && e.getTargetSizes().empty());

          // replace repartition with new extractslice
          auto nES = builder.create<::imex::dist::ExtractSliceOp>(
              e.getLoc(), o.getResult().getType(), combined, e.getOffsets(),
              e.getSizes(), e.getStrides(), tOffs, tSizes);
          o->replaceAllUsesWith(::mlir::ValueRange{nES.getResult()});

          // replace all extract uses as well
          // FIXME: uses of orig extractslice might be before first repartition
          //        we'll get a compiler error in that case
          e->replaceAllUsesWith(::mlir::ValueRange{nES.getResult()});
        }

        for (auto o : _rpops) {
          o->erase();
        }
        for (auto o : _extracts) {
          o->erase();
        }
      }
    }
  }
};
#endif

} // namespace
} // namespace dist

/// Create a pass to eliminate Dist ops
std::unique_ptr<::mlir::Pass> createDistCoalescePass() {
  return std::make_unique<::imex::dist::DistCoalescePass>();
}

} // namespace imex
