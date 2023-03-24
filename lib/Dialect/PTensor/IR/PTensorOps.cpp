//===- PTensor.cpp - PTensor dialect  --------------------------*- C++ -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the PTensor dialect and its basic operations.
///
//===----------------------------------------------------------------------===//

#include <imex/Dialect/Dist/Utils/Utils.h>
#include <imex/Dialect/PTensor/IR/PTensorOps.h>
#include <imex/Utils/PassUtils.h>
#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/Utils/StaticValueUtils.h>
#include <mlir/IR/DialectImplementation.h>

namespace imex {
namespace ptensor {

void PTensorDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
      >();
  addOperations<
#define GET_OP_LIST
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>
      >();
}

} // namespace ptensor
} // namespace imex

namespace imex {
namespace ptensor {

::mlir::MemRefType PTensorType::getMemRefType() {
  return ::imex::getMemRefType(getContext(), getShape(), getElementType());
}

::mlir::RankedTensorType PTensorType::getTensorType() {
  return ::imex::getTensorType(getContext(), getShape(), getElementType());
}

} // namespace ptensor
} // namespace imex

bool imex::ptensor::PTensorBase::hasRank() const { return true; }

llvm::ArrayRef<int64_t> imex::ptensor::PTensorBase::getShape() const {
  return cast<PTensorType>().getShape();
}

imex::ptensor::PTensorBase imex::ptensor::PTensorBase::cloneWith(
    llvm::Optional<llvm::ArrayRef<int64_t>> shape, Type elementType) const {
  auto t = cast<PTensorType>();
  return PTensorType::get(shape.value_or(getShape()), elementType,
                          t.getEnvironment(), t.getLayout());
}

bool imex::ptensor::PTensorBase::isValidElementType(Type type) {
  return type.isIntOrIndexOrFloat();
}

static mlir::LogicalResult parseShape(mlir::AsmParser &parser,
                                      llvm::SmallVector<int64_t> &shape,
                                      mlir::Type &type) {
  llvm::SmallVector<int64_t> dimensions;
  if (parser.parseDimensionList(dimensions))
    return mlir::failure();

  mlir::Type t;
  if (parser.parseType(t))
    return mlir::failure();

  shape = std::move(dimensions);
  type = std::move(t);
  return mlir::success();
}

static void printShape(mlir::AsmPrinter &printer, llvm::ArrayRef<int64_t> shape,
                       mlir::Type type) {
  for (int64_t dim : shape) {
    if (mlir::ShapedType::isDynamic(dim))
      printer << '?';
    else
      printer << dim;
    printer << 'x';
  }
  printer << type;
}

#include <imex/Dialect/PTensor/IR/PTensorOpsDialect.cpp.inc>
#define GET_TYPEDEF_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOpsTypes.cpp.inc>
#define GET_OP_CLASSES
#include <imex/Dialect/PTensor/IR/PTensorOps.cpp.inc>
