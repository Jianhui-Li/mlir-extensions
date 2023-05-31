//===- PTensorOps.h - PTensor dialect  --------------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the PTensor dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _PTENSOR_OPS_TD_INCLUDED_
#define _PTENSOR_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ShapedOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// Provide a definition of the 'PTensor' dialect in the ODS framework so that we
// can define our operations.
def PTensor_Dialect : Dialect {
    // The namespace of our dialect
    let name = "ptensor";

    // A short one-line summary of our dialect.
    let summary = "A high-level dialect for parallel tensor operations";

    // A longer description of our dialect.
    let description = [{
            The ptensor dialect describes parallel operations on tensors.
            Generic parallel patterns are provided, such as element-wise-unary,
            element-wise-binary or reduce.

            Generally the PTensor dialect is intended to provide high-level abstractions
            to allow compute-follows-data semantics. For this the PTensorType annotates
            RankedTensors with information about the location (device) of
            the tensor-data when PTensors are created.

            Initially the functional scope of the dialect is the
            [array-API](https://data-apis.org/array-api/latest/index.html).
        }];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::ptensor";

    //  We use the default parser/printer which handles registered types
    let useDefaultTypePrinterParser = true;
}

// common base classes for types in PTensor dialect
class PTensor_Type<string name, string typeMnemonic, list<Trait> traits = [],
                   string baseCppClass = "::mlir::Type">
    : TypeDef<PTensor_Dialect, name, traits, baseCppClass> {
  let mnemonic = typeMnemonic;
}

def PTensor_PTensor : PTensor_Type<"PTensor", "ptensor", [ShapedTypeInterface],
  "::imex::ptensor::PTensorBase"> {
  let summary = "Multi-dimensional numpy-like array";
  let description = [{
    Multi-dimensional numpy-like array.

    Contrary to upstream tensor type is has a reference semantics and allow to
    modify data inplace.

    Has optional `environment` attribute, which specifies additional environment
    information for computations on this tensor (e.g. it computation must run on
    specific GPU device).

    TBD
  }];
  let parameters = (ins
    ArrayRefParameter<"int64_t">:$shape,
    "::mlir::Type":$elementType,
    OptionalParameter<"::mlir::Attribute">:$environment,
    OptionalParameter<"::mlir::StringAttr">:$layout
  );

  let builders = [
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType,
      CArg<"::mlir::Attribute", "{}">:$environment,
      CArg<"::llvm::Optional<::llvm::StringRef>", "std::nullopt">:$layout
    ), [{
      auto ctx = elementType.getContext();
      auto l = layout ? ::mlir::StringAttr::get(ctx, *layout) : ::mlir::StringAttr{};
      return $_get(ctx, shape, elementType, environment, l);
    }]>,
    TypeBuilderWithInferredContext<(ins
      "::llvm::ArrayRef<int64_t>":$shape,
      "::mlir::Type":$elementType,
      "::mlir::Attribute":$environment,
      "::mlir::StringAttr":$layout
    ), [{
      auto ctx = elementType.getContext();
      return $_get(ctx, shape, elementType, environment, layout);
    }]>
  ];

  let extraClassDeclaration = [{
    ::mlir::MemRefType getMemRefType();
    ::mlir::RankedTensorType getTensorType();
    using ::mlir::ShapedType::Trait<PTensorType>::clone;
    using ::mlir::ShapedType::Trait<PTensorType>::getElementTypeBitWidth;
    using ::mlir::ShapedType::Trait<PTensorType>::getRank;
    using ::mlir::ShapedType::Trait<PTensorType>::getNumElements;
    using ::mlir::ShapedType::Trait<PTensorType>::isDynamicDim;
    using ::mlir::ShapedType::Trait<PTensorType>::hasStaticShape;
    using ::mlir::ShapedType::Trait<PTensorType>::getNumDynamicDims;
    using ::mlir::ShapedType::Trait<PTensorType>::getDimSize;
    using ::mlir::ShapedType::Trait<PTensorType>::getDynamicDimIndex;
  }];

  let assemblyFormat = "`<` custom<Shape>($shape, $elementType) (`:` $layout^)? (`,` $environment^)? `>`";
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class PTensor_Op<string mnemonic, list<Trait> traits = []> :
    Op<PTensor_Dialect, mnemonic, traits>;


def MkPTensorOp : PTensor_Op<"init_ptensor", [Pure]> {
    let summary = "Create a new PTensor from tensor, device info";
    let description = [{
        High-level operation to create a new PTensor.
        Every PTensor is expected to be created by this operation.
        FIXME The resulting PTensor will be marked as living on a device if $device is not none.
    }];

    let arguments = (ins AnyRankedTensor:$tensor,
                         AnyType:$device);
    let results = (outs PTensor_PTensor:$res);
    let skipDefaultBuilders = 1;
    let builders = [
      OpBuilder<(ins "bool":$onDevice,
                     "::mlir::Value":$tensor,
                     "::mlir::Value":$device), [{
        $_state.addOperands({tensor, device});
        auto mr = tensor.getType().dyn_cast<::mlir::RankedTensorType>();
        $_state.addTypes(::imex::ptensor::PTensorType::get(
            mr.getShape(),
            mr.getElementType())); // FIXME device
      }]>,
      // Default: no device
      OpBuilder<(ins "::mlir::Value":$tensor), [{
        auto dmy = createInt<1>($_builder.getUnknownLoc(), $_builder, 0);
        $_state.addOperands({tensor, dmy});
        auto mr = tensor.getType().dyn_cast<::mlir::RankedTensorType>();
        $_state.addTypes(::imex::ptensor::PTensorType::get(
            mr.getShape().size(),
            mr.getElementType()));
      }]>,
    ];
}

def ExtractRawPtrOp : PTensor_Op<"extract_raw_ptr", [Pure]> {
    let summary = "Extract the raw memref pointer from a given PTensor";
    let description = [{
        High-level operation to extract the raw pointer of a MemRef from the given PTensor.
        Assumes that the given PTensor was created by MkPTensorOp.
    }];

    let arguments = (ins AnyType:$source);
    let results = (outs Index:$raw_pointer);

    let builders = [
        OpBuilder<(ins "::mlir::Value":$tnsr), [{
            build($_builder, $_state, $_builder.getIndexType(), tnsr);
        }]>,
    ];
}

def ExtractTensorOp : PTensor_Op<"extract_tensor", [Pure]> {
    let summary = "Extract Tensor from a PTensor";
    let description = [{
        High-level operation to extract the Tensor from the given PTensor.
        Assumes that the given PTensor was created by MkPTensorOp.
    }];

    let arguments = (ins AnyType:$input);
    let results = (outs AnyType);

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$tnsr), [{
            auto mrtyp = tnsr.getType().dyn_cast<::imex::ptensor::PTensorType>().getTensorType();
            assert(mrtyp);
            build($_builder, $_state, mrtyp, tnsr);
        }]>,
    ];
}

def DimOp : PTensor_Op<"dim", [
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
    ConditionallySpeculatable, NoMemoryEffect,
    ShapedDimOpInterface]> {
  let summary = "dimension index operation";
  let description = [{
    The `dim` operation takes a array and a dimension operand of type `index`.
    It returns the size of the requested dimension of the given array.
    If the dimension index is out of bounds the behavior is undefined.
  }];

  let arguments = (ins AnyType:$source,
                       Index:$index);
  let results = (outs Index:$result);

  let assemblyFormat = [{
    $source $index attr-dict `:` qualified(type($source)) `->` qualified(type($result))
  }];

  let builders = [
    OpBuilder<(ins "::mlir::Value":$source, "int64_t":$index)>,
  ];

  let extraClassDeclaration = [{
    /// Helper function to get the index as a simple integer if it is constant.
    ::llvm::Optional<int64_t> getConstantIndex();

    /// Interface method of ShapedDimOpInterface: Return the source tensor.
    ::mlir::Value getShapedValue() { return getSource(); }

    /// Interface method of ShapedDimOpInterface: Return the dimension.
    ::mlir::OpFoldResult getDimension() { return getIndex(); }

    /// Interface method for ConditionallySpeculatable.
    ::mlir::Speculation::Speculatability getSpeculatability();
  }];

  // let hasVerifier = 1;
  // let hasFolder = 1;
  let hasCanonicalizer = 1;
}

// Base class for ops with static/dynamic offset, sizes and strides
// attributes/arguments.
class PTensor_OpWithOffsetSizesAndStrides<string mnemonic,
                                          list<Trait> traits = []>
    : PTensor_Op<mnemonic, traits> {
  code extraBaseClassDeclaration = [{
    /// Returns the dynamic sizes for this subview operation if specified.
    ::mlir::Operation::operand_range getDynamicSizes() { return getSizes(); }

    /// Return the list of Range (i.e. offset, size, stride). Each
    /// Range entry contains either the dynamic value or a ConstantIndexOp
    /// constructed with `b` at location `loc`.
    ::mlir::SmallVector<::mlir::Range, 8> getOrCreateRanges(
        ::mlir::OpBuilder &b, ::mlir::Location loc) {
      return ::mlir::getOrCreateRanges(*this, b, loc);
    }
  }];
}

def SubviewOp : PTensor_OpWithOffsetSizesAndStrides<"subview", [
    Pure, AttrSizedOperandSegments,
    DeclareOpInterfaceMethods<ReifyRankedShapedTypeOpInterface>,
    OffsetSizeAndStrideOpInterface,
    ViewLikeOpInterface
  ]> {
  let summary = "array subview operation";
  let description = [{
    The "subview" operation converts a array type to another array type
    which represents a reduced-size view of the original array as specified by
    the operation's offsets, sizes and strides arguments.
  }];

  let arguments = (ins
    AnyType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    DenseI64ArrayAttr:$static_offsets,
    DenseI64ArrayAttr:$static_sizes,
    DenseI64ArrayAttr:$static_strides
  );
  let results = (outs PTensor_PTensor:$result);

  let assemblyFormat = [{
    $source ``
    custom<DynamicIndexList>($offsets, $static_offsets)
    custom<DynamicIndexList>($sizes, $static_sizes)
    custom<DynamicIndexList>($strides, $static_strides)
    attr-dict `:` qualified(type($source)) `to` qualified(type($result))
  }];

  let builders = [
    // Build a SubViewOp with mixed static and dynamic entries and custom
    // result type. If the type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with mixed static and dynamic entries and inferred
    // result type.
    OpBuilder<(ins
      "PTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$offsets,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$sizes,
      "::mlir::ArrayRef<::mlir::OpFoldResult>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with static entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with static entries and inferred result type.
    OpBuilder<(ins
      "PTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ArrayRef<int64_t>":$offsets,
      "::mlir::ArrayRef<int64_t>":$sizes,
      "::mlir::ArrayRef<int64_t>":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with dynamic entries and custom result type. If the
    // type passed is nullptr, it is inferred.
    OpBuilder<(ins
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>,

    // Build a SubViewOp with dynamic entries and inferred result type.
    OpBuilder<(ins
      "PTensorType":$resultType,
      "::mlir::Value":$source,
      "::mlir::ValueRange":$offsets,
      "::mlir::ValueRange":$sizes,
      "::mlir::ValueRange":$strides,
      CArg<"::mlir::ArrayRef<::mlir::NamedAttribute>", "{}">:$attrs)>
  ];

  let extraClassDeclaration = extraBaseClassDeclaration # [{
    /// Returns the type of the base tensor operand.
    PTensorType getSourceType();

    /// The result of an subview is always a tensor.
    PTensorType getType() {
      return getResult().getType().cast<PTensorType>();
    }

    /// Compute the rank-reduction mask that can be applied to map the source
    /// tensor type to the result tensor type by dropping unit dims.
    llvm::Optional<llvm::SmallDenseSet<unsigned>>
    computeRankReductionMask() {
      return ::mlir::computeRankReductionMask(getSourceType().getShape(),
                                              getType().getShape());
    };

    /// An subview result type can be inferred, when it is not
    /// rank-reduced, from the source type and the static representation of
    /// offsets, sizes and strides. Special sentinels encode the dynamic case.
    static PTensorType inferResultType(
      PTensorType sourceType,
      ::mlir::ArrayRef<int64_t> staticOffsets,
      ::mlir::ArrayRef<int64_t> staticSizes,
      ::mlir::ArrayRef<int64_t> staticStrides);
    static PTensorType inferResultType(
      PTensorType sourceType,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticOffsets,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticSizes,
      ::mlir::ArrayRef<::mlir::OpFoldResult> staticStrides);

    /// A rank-reducing result type can be inferred from the desired result
    /// shape. Only the layout map is inferred.
    ///
    /// Note: The result shape cannot be inferred with just the result rank and
    /// and the desired sizes. In case there are more "ones" among the sizes
    /// than the difference in source/result rank, it is not clear which dims of
    /// size one should be dropped.
    static PTensorType inferRankReducedResultType(::mlir::ArrayRef<int64_t> resultShape,
                                                  PTensorType sourceType,
                                                  ::mlir::ArrayRef<int64_t> staticOffsets,
                                                  ::mlir::ArrayRef<int64_t> staticSizes,
                                                  ::mlir::ArrayRef<int64_t> staticStrides);
    static PTensorType inferRankReducedResultType(::mlir::ArrayRef<int64_t> resultShape,
                                                  PTensorType sourceType,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticOffsets,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticSizes,
                                                  ::mlir::ArrayRef<::mlir::OpFoldResult> staticStrides);

    /// Return the expected rank of each of the`static_offsets`, `static_sizes`
    /// and `static_strides` attributes.
    std::array<unsigned, 3> getArrayAttrMaxRanks() {
      unsigned rank = getSourceType().getRank();
      return {rank, rank, rank};
    }

    /// Return the number of leading operands before the `offsets`, `sizes` and
    /// and `strides` operands.
    static unsigned getOffsetSizeAndStrideStartOperandIndex() { return 1; }

    /// Return the dimensions of the source that are dropped in the
    /// result when the result is rank-reduced.
    ::llvm::SmallBitVector getDroppedDims();

    ::mlir::Value getViewSource() { return getSource(); }
  }];

  // let hasFolder = 1;
}


def InsertSliceOp : PTensor_Op<"insert_slice", [SameVariadicOperandSize]> {
  let summary = "Copy values from a tensor into a slice of another.";
  let description = [{
    Copy values from a tensor into a slice of another.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides
  );

  let assemblyFormat = [{
    $source `into` $destination `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($source)) `into` qualified(type($destination))
  }];
}

def LoadOp : PTensor_Op<"load",
     [TypesMatchWith<"result type matches element type of 'array'",
                     "array", "result",
                     "$_self.cast<PTensorType>().getElementType()">]> {
  let summary = "array element load operation";
  let description = [{
    The `load` op reads an element from an array specified by an index list. The
    output of load is a new value with the same type as the elements of the
    array. The arity of indices is the rank of the array (i.e., if the array
    loaded from is of rank 3, then 3 indices are required for the load following
    the array identifier).
  }];

  let arguments = (ins AnyType:$array,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, CArg<"::mlir::ValueRange", "{}">:$indices), [{
      auto arrayType = array.getType().cast<PTensorType>();
      $_state.addOperands(array);
      $_state.addOperands(indices);
      $_state.types.push_back(arrayType.getElementType());
    }]>];

  let assemblyFormat = "$array `[` $indices `]` attr-dict `:` qualified(type($array))";

  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}

def LinSpaceOp : PTensor_Op<"linspace", [AttrSizedOperandSegments]> {
    let summary = "Returns evenly spaced numbers over a specified interval.";
    let description = [{
        Number of of generated values is either num or num+1 depending on whether endpoint is True or False, respectively.
        See Array API.

        Optionally assigns it to device `device`, and team `team`.
    }];

    let arguments = (ins AnyType:$start, AnyType:$stop, AnyType:$num, UnitAttr:$endpoint,
                         Optional<AnyType>:$device, Optional<AnyType>:$team);
    // result is a ptensor
    let results = (outs PTensor_PTensor);

    let assemblyFormat = [{
      $start $stop $num (`true` $endpoint^):(`false`)? oilist(`device` $device | `team` $team) attr-dict `:` `(` type(operands) `)` `->` qualified(type(results))
    }];
}

def CreateOp : PTensor_Op<"create", [AttrSizedOperandSegments]> {
    let summary = "Returns a new PTensor having a specified shape and type and optionally filled with a value.";
    let description = [{
        Returns a new PTensor having a specified shape and type and optionally filled with a value.

        Optionally assigns it to device `device`, and team `team`.
    }];

    let arguments = (ins Variadic<Index>:$shape, I8Attr:$dtype, Optional<AnyType>:$value,
                         Optional<AnyType>:$device, Optional<AnyType>:$team);
    // result is a ptensor
    let results = (outs PTensor_PTensor);

    let assemblyFormat = [{
      $shape oilist(`value` $value | `device` $device | `team` $team) attr-dict `:` `(` type(operands) `)` `->` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::ValueRange":$shape, "::imex::ptensor::DType":$dtype, "::mlir::Value":$value, "::mlir::Value":$device, "::mlir::Value":$team), [{
            auto dt = toMLIR($_builder, dtype);
            build($_builder, $_state,
                  ::imex::ptensor::PTensorType::get(getShapeFromValues(shape), dt), // FIXME device
                  shape, $_builder.getI8IntegerAttr(dtype), value, device, team);
        }]>,
    ];

    let extraClassDeclaration = [{
      ::imex::ptensor::DType getDType() {
        return static_cast<::imex::ptensor::DType>(getDtype());
      }
    }];
}

def ReshapeOp : PTensor_Op<"reshape", []> {
    let summary = "Reshapes an array without changing its data.";
    let description = [{
        Reshapes an array without changing its data. Memory is re-used as requested.
        See Array API.
    }];

    let arguments = (ins AnyType:$src, Variadic<Index>:$shape, OptionalAttr<I1Attr>:$copy);
    let results = (outs AnyType);
}

def EWBinOp : PTensor_Op<"ewbin", []> {
    let summary = "Apply elementwise binary operation";
    let description = [{
        Apply the `op(lhs[i], rhs[i])` on all elements `i` and return a new ptensor.
        The broadcasting rules of the array-API are applied to operator and result types.
    }];

    // ewbin takes 2 PTensorType operands: lhs and rhs
    let arguments = (ins AnyAttr:$op, AnyType:$lhs, AnyType:$rhs);
    // result is a ptensor
    let results = (outs AnyType);
}

def EWUnyOp : PTensor_Op<"ewuny", []> {
    let summary = "Apply elementwise unary operation";
    let description = [{
        Apply the `op(src[i])` on all elements `i` and return a new ptensor.
    }];

    // ewuny takes 1 operand (PTensorType) and one attribute (unary operation)
    let arguments = (ins AnyAttr:$op, AnyType:$src);
    // result is a ptensor
    let results = (outs AnyType);
}

def ReductionOp : PTensor_Op<"reduction", []> {
    let summary = "Apply reduction operation";
    let description = [{
        Apply the reduction operation `op` over all elements of `input`.
        The produced result is a 0-dim tensor with the same dtype as `input`.
    }];

    // reduction takes 1 operand (PTensorType) and one attribute (reduction operation)
    let arguments = (ins AnyAttr:$op, AnyType:$input);
    // result is a ptensor
    let results = (outs PTensor_PTensor);
}

#endif // _PTENSOR_OPS_TD_INCLUDED_
