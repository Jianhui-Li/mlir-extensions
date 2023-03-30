//===- Utils.h - Utils for PTensor dialect  -----------------------*- C++
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
/// This file declares the utils for the ptensor dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensor_UTILS_H_INCLUDED_
#define _PTensor_UTILS_H_INCLUDED_

#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>
#include <mlir/Dialect/Linalg/Utils/Utils.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/Builders.h>

namespace imex {

// *******************************
// ***** Some helper functions ***
// *******************************

namespace ptensor {

inline ::mlir::Type toMLIR(::mlir::OpBuilder &b, DType dt) {
  switch (dt) {
  case F64:
    return b.getF64Type();
  case F32:
    return b.getF32Type();
  case I64:
    return b.getI64Type();
  case U64:
    return b.getI64Type();
  case I32:
    return b.getI32Type();
  case U32:
    return b.getI32Type();
  case I16:
    return b.getI16Type();
  case U16:
    return b.getI16Type();
  case I8:
    return b.getI8Type();
  case U8:
    return b.getI8Type();
  case I1:
    return b.getI1Type();
  default:
    assert(!"Cannot handle unknown DType");
  };
  return {};
}

inline DType fromMLIR(const ::mlir::Type &typ) {
  if (typ.isF64())
    return F64;
  else if (typ.isF32())
    return F32;
  else if (typ.isIntOrIndex()) {
    auto w = typ.getIntOrFloatBitWidth();
    auto u = !typ.isIndex() && typ.isUnsignedInteger();
    switch (w) {
    case 64:
      return u ? U64 : I64;
    case 32:
      return u ? U32 : I32;
    case 16:
      return u ? U16 : I16;
    case 8:
      return u ? U8 : I8;
    case 1:
      return I1;
    };
  }
  assert(!"Type not supported by PTensor");
}

inline ::mlir::Value createDType(::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder, ::mlir::Type mt) {
  return createInt<sizeof(int) * 8>(
      loc, builder, static_cast<int>(::imex::ptensor::fromMLIR(mt)));
}

inline ::mlir::Value createDType(::mlir::Location &loc,
                                 ::mlir::OpBuilder &builder,
                                 ::mlir::MemRefType mrt) {
  return createDType(loc, builder, mrt.getElementType());
}

inline auto createGetLocalSizes(::mlir::Location loc,
                                ::mlir::OpBuilder &builder,
                                ::mlir::Value lPTnsr) {
  auto PTTyp = lPTnsr.getType().dyn_cast<::imex::ptensor::PTensorType>();
  assert(PTTyp);
  auto lTensor = builder.create<::imex::ptensor::ExtractTensorOp>(loc, lPTnsr);
  auto rank = PTTyp.getRank();
  ::mlir::SmallVector<::mlir::Value> dims(rank);

  for (int64_t i = 0; i < rank; ++i) {
    dims[i] = ::mlir::linalg::createOrFoldDimOp(builder, loc, lTensor, i);
  }

  return dims;
}

// template <int W, typename T>
// ::mlir::Value createSignlessInt(::mlir::OpBuilder &b,
//                                 const ::mlir::Location &loc, T val) {
//   return b
//       .create<::mlir::arith::ConstantOp>(
//           loc, b.getIntegerAttr(b.getIntegerType(W), val))
//       .getResult();
// }

} // namespace ptensor
} // namespace imex

#endif //  _PTensor_UTILS_H_INCLUDED_
