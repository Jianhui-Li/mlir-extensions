//===-- Passes.td - Dist pass definition file --------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines passes/transformations of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_PASSES_TD_INCLUDED_
#define _Dist_PASSES_TD_INCLUDED_

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// DistCoalesce
//===----------------------------------------------------------------------===//

def DistCoalesce : Pass<"dist-coalesce", "::mlir::func::FuncOp"> {
  let summary = "Coalesce operations from Dist dialect.";
  let description = [{
    TODO
  }];
  let constructor = "imex::createDistCoalescePass()";
  let dependentDialects = ["::imex::dist::DistDialect",
                           "::mlir::arith::ArithDialect",
                           "::mlir::tensor::TensorDialect",
                           "::mlir::memref::MemRefDialect"];
  let options = [
    Option<"in_jit", "in-jit", "bool", /*default=*/"true",
           "Assume (or not) that pass is run within a jit.">,
  ];
}

#endif // _Dist_PASSES_TD_INCLUDED_
