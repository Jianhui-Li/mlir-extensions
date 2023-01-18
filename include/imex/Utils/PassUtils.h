//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2022 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines utility functions for writing passes.
//
//===----------------------------------------------------------------------===//

#ifndef _IMEX_PASSUTILS_H_
#define _IMEX_PASSUTILS_H_

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/IR/Dominance.h>
#include <mlir/Pass/Pass.h>

#include <iostream>

namespace imex {

/// @return get ::mlir::FloatAttr with given Value and bitwidth W
template <int W = 64, typename T = double>
::mlir::FloatAttr getFloatAttr(::mlir::OpBuilder &builder, T val) {
  if (W == 64)
    return builder.getF64FloatAttr(val);
  if (W == 32)
    return builder.getF32FloatAttr(val);
  assert(!"only 32- and 64-bit floats supported");
}

/// @return new float ::mlir::Value with given Value and bitwidth W
template <int W = 64, typename T = double>
::mlir::Value createFloat(const ::mlir::Location &loc,
                          ::mlir::OpBuilder &builder, T val) {
  auto attr = getFloatAttr<W>(builder, val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return get ::mlir::IntegerAttr with given Value and bitwidth W
template <int W = 64>
::mlir::IntegerAttr getIntAttr(::mlir::OpBuilder &builder, int64_t val) {
  return builder.getIntegerAttr(builder.getIntegerType(W), val);
}

/// @return new integer ::mlir::Value with given Value and bitwidth W
template <int W = 64>
::mlir::Value createInt(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                        int64_t val) {
  auto attr = getIntAttr<W>(builder, val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

/// @return new index ::mlir::Value with given Value
inline ::mlir::Value createIndex(const ::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, int64_t val) {
  auto attr = builder.getIndexAttr(val);
  return builder.create<::mlir::arith::ConstantOp>(loc, attr);
}

inline ::mlir::Value createIndexCast(const ::mlir::Location &loc,
                                     ::mlir::OpBuilder &builder,
                                     ::mlir::Value val,
                                     ::mlir::Type intTyp = ::mlir::Type()) {
  if (!intTyp)
    intTyp = builder.getIndexType();
  return val.getType() == intTyp
             ? val
             : builder.create<::mlir::arith::IndexCastOp>(loc, intTyp, val)
                   .getResult();
}

/// get dyn-sized mlir::RankedTensorType for given rank and elType
inline auto getTensorType(::mlir::MLIRContext *ctxt, int64_t rank,
                          ::mlir::Type elType) {
  return ::mlir::RankedTensorType::get(
      std::vector<int64_t>(rank, ::mlir::ShapedType::kDynamic),
      elType); //, layout);
}

/// create an empty RankedTensor with tiven shape and elType
inline auto createEmptyTensor(::mlir::OpBuilder &builder, ::mlir::Location loc,
                              ::mlir::Type elType, ::mlir::ValueRange shp) {
  return builder.createOrFold<::mlir::tensor::EmptyOp>(
      loc, getTensorType(builder.getContext(), shp.size(), elType), shp);
}

/// get dyn-sized mlir::RankedTensorType for given rank and elType
/// if strided==true make it a strided layout
inline auto getMemRefType(::mlir::MLIRContext *ctxt, int64_t rank,
                          ::mlir::Type elType, bool strided = true) {
  static auto kDynamic = ::mlir::ShapedType::kDynamic;
  auto layout = ::mlir::StridedLayoutAttr::get(
      ctxt, kDynamic, ::mlir::SmallVector<int64_t>(rank, kDynamic));
  return ::mlir::MemRefType::get(std::vector<int64_t>(rank, kDynamic), elType,
                                 strided ? layout
                                         : ::mlir::StridedLayoutAttr{});
}

/// Create a 1d MemRef alloc with given size and elType
inline auto createAllocMR(::mlir::OpBuilder &builder, ::mlir::Location loc,
                          ::mlir::Type elType, int64_t sz) {
  return builder.create<::mlir::memref::AllocOp>(
      loc, ::mlir::MemRefType::get({sz}, elType), builder.getI64IntegerAttr(8));
}

/// Create a 1d MemRef from given elements and elType
inline ::mlir::Value createMemRefFromElements(::mlir::OpBuilder &builder,
                                              ::mlir::Location loc,
                                              ::mlir::Type elType,
                                              ::mlir::ValueRange elts) {
  int64_t N = elts.size();
  auto mr = createAllocMR(builder, loc, elType, N);
  for (auto i = 0; i < N; ++i) {
    auto idx = createIndex(loc, builder, i);
    (void)builder.create<::mlir::memref::StoreOp>(loc, elts[i], mr, idx);
  }
  return mr;
}

/// @return members of given 1d memref as individual values
inline auto createValuesFromMemRef(::mlir::OpBuilder &builder,
                                   ::mlir::Location loc, ::mlir::Value mr) {
  auto mrTyp = mr.getType().dyn_cast<::mlir::MemRefType>();
  assert(mrTyp && mrTyp.getShape().size() == 1);
  auto rank = mrTyp.getShape()[0];
  ::mlir::SmallVector<::mlir::Value> vals(rank);
  for (auto i = 0; i < rank; ++i) {
    auto _i = createIndex(loc, builder, i);
    vals[i] = builder.create<::mlir::memref::LoadOp>(loc, mr, _i).getResult();
  }
  return vals;
}

// when possible, move up operations of a certain type so that they are
// close together.
template <typename OP, typename GETINPUT>
void groupOps(::mlir::DominanceInfo &domA, ::mlir::Operation *root,
              GETINPUT getInput) {
  llvm::SmallVector<OP> ops;

  // Find all operations of type OP within root
  root->walk([&](::mlir::Operation *op) {
    if (auto typedOp = ::mlir::dyn_cast<OP>(op)) {
      ops.emplace_back(typedOp);
      return;
    }
  });

  llvm::SmallVector<OP> dominators;
  llvm::SmallVector<OP> dominators2;
  dominators2.emplace_back(ops.front());

  // we treat the first found op as the dominating operation
  // We try to move up all found ops to right after the dominator
  // Ops which cannot be be moved will serve as new dominators and we
  // recursively try to move remaining ops to them
  do {
    dominators.swap(
        dominators2); // new dominators will get stored in dominators2
    for (auto dominator : dominators) {
      auto iPnt = dominator;
      for (auto op : ops) {
        if (domA.properlyDominates(dominator, op, false)) {
          auto dTnsr = getInput(op);
          auto defOp = dTnsr.getDefiningOp();
          if (domA.properlyDominates(defOp, dominator)) {
            op->moveAfter(iPnt);
            iPnt = op;
          } else {
            dominators2.emplace_back(op);
          }
        }
      }
    }
    dominators.clear();
  } while (!dominators2.empty());
}

} // namespace imex
#endif // _IMEX_PASSUTILS_H_
