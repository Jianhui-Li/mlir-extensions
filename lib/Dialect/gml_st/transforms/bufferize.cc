/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file implements logic for translating mixed IR to buffer form.
// Currently it supports MHLO and some operations from the Standard dialect.

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

#include "imex/Dialect/gml_st/IR/gml_st_ops.h"
#include "imex/Dialect/gml_st/transforms/bufferizable_op_interface_impl.h"
#include "imex/Dialect/gml_st/transforms/passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotModuleBufferize.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Shape/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {

#define GEN_PASS_DEF_ONESHOTBUFFERIZE
#include "imex/Dialect/gml_st/transforms/passes.h.inc"

namespace {

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for bufferization.

static Value materializeToTensor(OpBuilder &builder, TensorType type,
                                 ValueRange inputs, Location loc) {
  assert(inputs.size() == 1);
  assert(inputs[0].getType().isa<BaseMemRefType>());
  return builder.create<bufferization::ToTensorOp>(loc, type, inputs[0]);
}

// TODO(pifon): Remove as soon as https://reviews.llvm.org/D93126 is landed.
class CustomBufferizeTypeConverter
    : public bufferization::BufferizeTypeConverter {
public:
  CustomBufferizeTypeConverter() {
    // Keep all types unchanged.
    addConversion([](Type type) { return type; });
    // Convert RankedTensorType to MemRefType.
    addConversion([](RankedTensorType type) -> Type {
      return MemRefType::get(type.getShape(), type.getElementType());
    });
    // Convert UnrankedTensorType to UnrankedMemRefType.
    addConversion([](UnrankedTensorType type) -> Type {
      return UnrankedMemRefType::get(type.getElementType(), 0);
    });
    addArgumentMaterialization(materializeToTensor);
    addSourceMaterialization(materializeToTensor);
    addTargetMaterialization([](OpBuilder &builder, BaseMemRefType type,
                                ValueRange inputs, Location loc) -> Value {
      assert(inputs.size() == 1);
      // Target materialization is invoked if the new operand type does not
      // match the expected type. A special case is when the new operand type is
      // a memref with a specified layout, i.e. non-empty affine map.
      // TODO(pifon) : Change how target materialization is invoked in dialect
      // conversion.
      if (auto memrefType = inputs[0].getType().dyn_cast<MemRefType>()) {
        assert(!memrefType.getLayout().isIdentity());
        return inputs[0];
      }
      assert(inputs[0].getType().isa<TensorType>());
      return builder.create<bufferization::ToMemrefOp>(loc, type, inputs[0]);
    });
  }
};

struct OneShotBufferizePass
    : public impl::OneShotBufferizeBase<OneShotBufferizePass> {
  // TODO(b/173201243): Move to tablegen.
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<bufferization::BufferizationDialect, linalg::LinalgDialect,
                    memref::MemRefDialect, scf::SCFDialect, shape::ShapeDialect,
                    vector::VectorDialect>();
    arith::registerBufferizableOpInterfaceExternalModels(registry);
    bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    gml_st::registerBufferizableOpInterfaceExternalModels(registry);
    linalg::registerBufferizableOpInterfaceExternalModels(registry);
    scf::registerBufferizableOpInterfaceExternalModels(registry);
    shape::registerBufferizableOpInterfaceExternalModels(registry);
    tensor::registerBufferizableOpInterfaceExternalModels(registry);
    vector::registerBufferizableOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override {
    bufferization::OneShotBufferizationOptions opts;
    opts.allowReturnAllocs = true;
    opts.bufferizeFunctionBoundaries = true;
    opts.functionBoundaryTypeConversion =
        bufferization::BufferizationOptions::LayoutMapOption::IdentityLayoutMap;
    opts.createDeallocs = false;
    opts.bufferAlignment = 64;

    ModuleOp module = getOperation();
    if (failed(bufferization::runOneShotModuleBufferize(module, opts))) {
      signalPassFailure();
    }
  }
};

} // namespace

namespace gml_st {
std::unique_ptr<OperationPass<ModuleOp>> createOneShotBufferizePass() {
  return std::make_unique<OneShotBufferizePass>();
}
} // namespace gml_st

} // namespace mlir
