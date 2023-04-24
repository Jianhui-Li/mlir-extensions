//===- PassUtils.h - Pass Utility Functions --------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
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
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
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

// cast between different scalar types
inline ::mlir::Value createCast(const ::mlir::Location &loc,
                                ::mlir::OpBuilder &builder, ::mlir::Value val,
                                ::mlir::Type dTyp) {
  auto vTyp = val.getType();
  assert(vTyp.isIntOrIndexOrFloat() && dTyp.isIntOrIndexOrFloat());
  if (vTyp == dTyp) {
    return val;
  } else if (vTyp.isIntOrIndex() && dTyp.isIntOrIndex()) {
    if ((vTyp.isIndex() || dTyp.isIndex())) {
      return createIndexCast(loc, builder, val, dTyp);
    } else if (vTyp.getIntOrFloatBitWidth() < dTyp.getIntOrFloatBitWidth()) {
      return builder.create<::mlir::arith::ExtSIOp>(loc, dTyp, val);
    } else {
      return builder.create<::mlir::arith::TruncIOp>(loc, dTyp, val);
    }
  } else if (vTyp.isIntOrIndex()) {
    val = createIndexCast(loc, builder, val, builder.getIntegerType(64));
    return builder.create<::mlir::arith::SIToFPOp>(loc, dTyp, val);
  } else if (dTyp.isIntOrIndex()) {
    if (dTyp == builder.getIndexType()) {
      val = builder.create<::mlir::arith::FPToSIOp>(
          loc, builder.getIntegerType(64), val);
      return createIndexCast(loc, builder, val, dTyp);
    } else {
      return builder.create<::mlir::arith::FPToSIOp>(loc, dTyp, val);
    }
  } else if (vTyp.getIntOrFloatBitWidth() < dTyp.getIntOrFloatBitWidth()) {
    return builder.create<::mlir::arith::ExtFOp>(loc, dTyp, val);
  }
  assert(!(vTyp.isIntOrIndex() || vTyp.isIntOrIndex()) &&
         vTyp.getIntOrFloatBitWidth() > dTyp.getIntOrFloatBitWidth());
  return builder.create<::mlir::arith::TruncFOp>(loc, dTyp, val);
}

/// @return array of static sizes form ValueRange: actual size if constant,
/// kDynamic if not
inline ::mlir::SmallVector<int64_t>
getShapeFromValues(const ::mlir::ValueRange &sizes) {
  auto rank = sizes.size();
  ::mlir::SmallVector<int64_t> szVec(rank, ::mlir::ShapedType::kDynamic);
  for (size_t i = 0; i < rank; ++i) {
    if (auto cval = ::mlir::getConstantIntValue(sizes[i]);
        cval && cval.value() == 1) {
      szVec[i] = cval.value();
    }
  }
  return szVec;
}

/// @return number of elements for given shape if all sizes are constant,
/// kDynamic otherwise
inline int64_t getSizeFromValues(const ::mlir::ValueRange &sizes) {
  int64_t sz = 0;
  for (auto s : sizes) {
    if (auto cval = ::mlir::getConstantIntValue(s)) {
      sz *= cval.value();
    } else {
      return ::mlir::ShapedType::kDynamic;
    }
  }
  return sz;
}

/// get dyn-sized mlir::RankedTensorType for given size values and elType
inline auto getTensorType(::mlir::MLIRContext *ctxt,
                          const ::mlir::ValueRange &sizes,
                          ::mlir::Type elType) {
  auto shape = getShapeFromValues(sizes);
  return ::mlir::RankedTensorType::get(shape, elType); //, layout);
}

/// get dyn-sized mlir::RankedTensorType for given shape and elType
inline auto getTensorType(::mlir::MLIRContext *ctxt,
                          ::mlir::ArrayRef<int64_t> shape,
                          ::mlir::Type elType) {
  return ::mlir::RankedTensorType::get(shape, elType); //, layout);
}

/// get dyn-sized mlir::RankedTensorType for given rank and elType
inline auto getTensorType(::mlir::MLIRContext *ctxt, int64_t rank,
                          ::mlir::Type elType) {
  return ::mlir::RankedTensorType::get(
      std::vector<int64_t>(rank, ::mlir::ShapedType::kDynamic),
      elType); //, layout);
}

/// combine dynamic and static sizes (as used by SubviewOps) into a
/// single ValueRange (vecotr of values)
inline ::mlir::SmallVector<::mlir::Value>
getMixedAsValues(const ::mlir::Location &loc, ::mlir::OpBuilder &builder,
                 const ::mlir::ValueRange &dyns,
                 ::llvm::ArrayRef<int64_t> statics) {
  ::mlir::SmallVector<::mlir::Value> out;
  auto dyn = dyns.begin();
  for (auto s : statics) {
    out.emplace_back(::mlir::ShapedType::isDynamic(s)
                         ? *(dyn++)
                         : createIndex(loc, builder, s));
  }
  return out;
}

/// similar to mlir::decomposeMixedValues but converting const values tot
/// statics
inline void
dispatchIndexValues(::mlir::OpBuilder &builder, ::mlir::Location loc,
                    const ::mlir::ValueRange &sizes,
                    ::mlir::SmallVectorImpl<::mlir::Value> &dynamicVec,
                    ::mlir::SmallVectorImpl<int64_t> &staticVec) {
  for (auto v : sizes) {
    if (auto cval = ::mlir::getConstantIntValue(v); cval && cval.value() == 1) {
      staticVec.emplace_back(cval.value());
    } else {
      dynamicVec.emplace_back(createIndexCast(loc, builder, v));
      staticVec.emplace_back(::mlir::ShapedType::kDynamic);
    }
  }
}

/// create an empty RankedTensor with given shape and elType
inline auto createEmptyTensor(::mlir::OpBuilder &builder, ::mlir::Location loc,
                              ::mlir::Type elType,
                              const ::mlir::ValueRange &shp) {
  ::mlir::SmallVector<int64_t> staticSizes;
  ::mlir::SmallVector<mlir::Value> dynamicSizes;
  dispatchIndexValues(builder, loc, shp, dynamicSizes, staticSizes);
  return builder
      .create<::mlir::tensor::EmptyOp>(loc, staticSizes, elType, dynamicSizes)
      .getResult();
}

/// create an empty RankedTensor for given result tensor type and operand shapes
inline auto createEmptyTensor(::mlir::OpBuilder &builder, ::mlir::Location loc,
                              ::mlir::TensorType resType,
                              const ::mlir::ValueRange &operands) {
  ::mlir::SmallVector<::mlir::Value> dynDims(resType.getRank());
  auto elType = resType.getElementType();
  for (auto arg : operands) {
    auto operandTy = arg.getType().cast<::mlir::ShapedType>();
    for (int i = 0; i < operandTy.getRank(); i++) {
      if (operandTy.isDynamicDim(i) && !dynDims[i])
        dynDims[i] = builder.create<::mlir::tensor::DimOp>(loc, arg, i);
    }
  }
  ::mlir::SmallVector<::mlir::Value> filteredDims;
  for (auto value : dynDims) {
    if (value) {
      filteredDims.push_back(value);
    }
  }
  return createEmptyTensor(builder, loc, elType, filteredDims);
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

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
inline auto getMemRefType(::mlir::MLIRContext *ctxt,
                          ::mlir::ArrayRef<int64_t> sizes, ::mlir::Type elType,
                          bool strided = true) {
  auto rank = sizes.size();
  static auto kDynamic = ::mlir::ShapedType::kDynamic;
  auto layout = ::mlir::StridedLayoutAttr::get(
      ctxt, kDynamic, ::mlir::SmallVector<int64_t>(rank, kDynamic));

  return ::mlir::MemRefType::get(
      sizes, elType, strided ? layout : ::mlir::StridedLayoutAttr{});
}

/// get mixed statically and dynamically sized mlir::MemRefType for given sizes
/// and elType if strided==true make it a strided layout
inline auto getMemRefType(::mlir::MLIRContext *ctxt,
                          const ::mlir::ValueRange &sizes, ::mlir::Type elType,
                          bool strided = true) {
  return getMemRefType(ctxt, getShapeFromValues(sizes), elType, strided);
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
} // namespace imex

// FIXME
#include <imex/Utils/ArithUtils.h>

namespace imex {
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

inline auto createExtractPtrFromMemRef(::mlir::OpBuilder &builder,
                                       ::mlir::Location loc, ::mlir::Value mr) {
  auto meta = builder.create<::mlir::memref::ExtractStridedMetadataOp>(loc, mr);
  return createExtractPtrFromMemRef(builder, loc, mr, meta);
}

inline auto createExtractPtrFromMemRefFromValues(::mlir::OpBuilder &builder,
                                                 ::mlir::Location loc,
                                                 ::mlir::ValueRange elts) {
  auto mr =
      createMemRefFromElements(builder, loc, builder.getIndexType(), elts);
  return createExtractPtrFromMemRef(builder, loc, mr);
}

// when possible, move up operations of a certain type so that they are
// close together.
template <typename OP, typename SELECT, typename GETINPUTS,
          typename SINGULARIZE>
void groupOps(::mlir::DominanceInfo &domA, ::mlir::Operation *root,
              SELECT select, GETINPUTS getInputs,
              SINGULARIZE singularize = nullptr) {

  llvm::SmallVector<OP> dominators, dominators2;

  // Find all operations of type OP within root
  root->walk([&](OP op) {
    if (select(op)) {
      dominators.emplace_back(op);
      return;
    }
  });

  // we treat the first found op as the dominating operation
  // We try to move up all found ops to right after the dominator
  // Ops which cannot be be moved will serve as new dominators and we
  // recursively try to move remaining ops to them
  while (dominators.size() > 1) {
    auto dominator = dominators.front();
    auto iPnt = dominator;
    while (dominators.size() > 1) {
      auto op = dominators.pop_back_val();
      if (domA.properlyDominates(dominator, op, false)) {
        bool can_move = true;
        auto oprnds = getInputs(op);
        for (auto d : oprnds) {
          auto defOp = d.getDefiningOp();
          if (defOp && !domA.properlyDominates(defOp, dominator)) {
            can_move = false;
            break;
          }
        }
        if (can_move) {
          if constexpr (singularize != nullptr) {
            if (singularize(dominator, op)) {
              op->replaceAllUsesWith(dominator);
              op->erase();
              continue;
            }
          }
          op->moveAfter(iPnt);
          iPnt = op;
          if constexpr (singularize == nullptr) {
            continue;
          }
        }
      }
      // not dominated or not movable
      dominators2.emplace_back(op);
    }
    dominators.clear();
    dominators.swap(dominators2);
  }
}

inline void printValsAsMemRef(::mlir::Location loc, ::mlir::OpBuilder &builder,
                              ::mlir::ValueRange vals) {
  auto et = vals[0].getType();
  auto memrefType = ::mlir::UnrankedMemRefType::get(et, {});
  auto mr = createMemRefFromElements(builder, loc, et, vals);
  auto cmr = builder.create<mlir::memref::CastOp>(loc, memrefType, mr);
  if (et == builder.getIndexType()) {
    builder.create<::mlir::func::CallOp>(loc, "printMemrefInd",
                                         ::mlir::TypeRange(), cmr.getResult());
  } else if (et == builder.getI64Type()) {
    builder.create<::mlir::func::CallOp>(loc, "printMemrefI64",
                                         ::mlir::TypeRange(), cmr.getResult());
  } else {
    assert(false);
  }
}

} // namespace imex
#endif // _IMEX_PASSUTILS_H_
