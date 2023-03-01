//===- PTensorOps.h - PTensor dialect  -------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file declares the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#ifndef _PTensor_OPS_H_INCLUDED_
#define _PTensor_OPS_H_INCLUDED_

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/Dialect.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Types.h>
#include <mlir/Interfaces/ShapedOpInterfaces.h>
#include <mlir/Interfaces/ViewLikeInterface.h>

#include "PTensorDefs.h"

namespace imex {
namespace ptensor {

class PTensorBase : public mlir::Type,
                    public mlir::ShapedType::Trait<PTensorBase> {
public:
  using Type::Type;

  /// Returns the element type of this tensor type.
  mlir::Type getElementType() const;

  /// Returns if this type is ranked, i.e. it has a known number of dimensions.
  bool hasRank() const;

  /// Returns the shape of this tensor type.
  llvm::ArrayRef<int64_t> getShape() const;

  /// Clone this type with the given shape and element type. If the
  /// provided shape is `None`, the current shape of the type is used.
  PTensorBase cloneWith(llvm::Optional<llvm::ArrayRef<int64_t>> shape,
                        mlir::Type elementType) const;

  /// Return true if the specified element type is ok in a tensor.
  static bool isValidElementType(Type type);

  /// Methods for support type inquiry through isa, cast, and dyn_cast.
  static bool classof(Type type);

  /// Allow implicit conversion to ShapedType.
  operator mlir::ShapedType() const { return cast<mlir::ShapedType>(); }
};

} // namespace ptensor
} // namespace imex

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.h.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.h.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.h.inc>

#include <imex/Dialect/PTensor/Utils/Utils.h>

#endif // _PTensor_OPS_H_INCLUDED_
