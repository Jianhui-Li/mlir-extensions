//===- DistOps.td - Dist dialect  --------------------------*- tablegen -*-===//
//
// Copyright 2023 Intel Corporation
// Part of the IMEX Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines basic operations of the Dist dialect.
///
//===----------------------------------------------------------------------===//

#ifndef _Dist_OPS_TD_INCLUDED_
#define _Dist_OPS_TD_INCLUDED_

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Provide a definition of the 'Dist' dialect in the ODS framework so that we
// can define our operations.
def Dist_Dialect : Dialect {
    // The namespace of our dialect
    let name = "dist";

    // A short one-line summary of our dialect.
    let summary = "A high-level dialect for distributing PTensor operations";

    // A longer description of our dialect.
    let description = [{
        The dist dialect describes interfaces for interacting with
	    a runtime which handles distributed aspects of PTensor operations.
    }];

    let dependentDialects = [
        "::imex::ptensor::PTensorDialect"
    ];

    // The C++ namespace that the dialect class definition resides in.
    let cppNamespace = "::imex::dist";
    let useDefaultTypePrinterParser = 1;
}

// common base classes for types in Dist dialect
class Dist_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<Dist_Dialect, name, traits> {
    let mnemonic = typeMnemonic;
}

def Dist_Tensor : Dist_Type<"DistTensor", "dtensor">
{
  let summary = "A type used to bind distributed information to a PTensor";
  let description = [{
    A distributed PTensor needs information like offset and shape of local partition.
    The DistTensor type is used to define operations to carry and extract such information.
  }];
  let parameters = (ins "::imex::ptensor::PTensorType":$p_tensor_type);
  let assemblyFormat = "`<` $p_tensor_type `>`";

  let extraClassDeclaration = [{
    ::mlir::Type getElementType();
  }];
}

// Base class for dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dist_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dist_Dialect, mnemonic, traits>;

def RuntimePrototypesOp : Dist_Op<"runtime_prototypes"> {
    let summary = "Add function prototypes used for calling into distributed runtime";
}

def NProcsOp : Dist_Op<"nprocs", [Pure]> {
    let summary = "Number of processes for given team";
    let arguments = (ins AnyType:$team);
    let results = (outs Index);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getIndexType(), team);
        }]>,
    ];
}

def PRankOp : Dist_Op<"prank", [Pure]> {
    let summary = "Process rank in team";
    let arguments = (ins AnyType:$team);
    let results = (outs Index);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$team), [{
            build($_builder, $_state, $_builder.getIndexType(), team);
        }]>,
    ];
}

def InitDistTensorOp : Dist_Op<"init_dist_tensor", [SameVariadicOperandSize, Pure]> {
    let summary = "Bind a PTensor to distributed meta information";
    let description = [{
        The attached PTensor is the local partiton of the distributed PTensor.
        The distributed meta information about a new PTensor provides
          - the global shape
          - balanced flag
          - the process-local offsets
          - the distributed team
    }];
    let arguments = (ins AnyType:$p_tensor, AnyType:$team, IndexAttr:$balanced, Variadic<Index>:$g_shape, Variadic<Index>:$l_offsets);
    let results = (outs Dist_Tensor);

    let assemblyFormat = [{
        $p_tensor $team $balanced $g_shape oilist(`offsets` $l_offsets) attr-dict `:` qualified(type(operands)) `to` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$tensor, "::mlir::Value":$team, "int64_t":$balanced, "::mlir::ValueRange":$g_shape, "::mlir::ValueRange":$l_offsets), [{
            auto dtTyp = tensor.getType().dyn_cast<::imex::dist::DistTensorType>();
            auto ptTyp = dtTyp ? dtTyp.getPTensorType() : tensor.getType().dyn_cast<::imex::ptensor::PTensorType>();
            assert(ptTyp);
            assert(l_offsets.empty() || l_offsets.size() == g_shape.size());
            build($_builder, $_state,
                  ::imex::dist::DistTensorType::get($_builder.getContext(), ptTyp),
                  tensor, team, $_builder.getIndexAttr(balanced), g_shape, l_offsets);
        }]>,
    ];
}

def GlobalShapeOfOp : Dist_Op<"global_shape_of", []> {
    let summary = "Get global shape of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs Variadic<Index>:$g_shape);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        auto rank = d_tensor.getType().dyn_cast<::imex::dist::DistTensorType>().getPTensorType().getRank();
        auto IndexType = $_builder.getIndexType();
        ::mlir::SmallVector<::mlir::Type> rt(rank, IndexType);
        build($_builder, $_state, ::mlir::TypeRange(rt), d_tensor);
      }]>,
    ];
}

def LocalOffsetsOfOp : Dist_Op<"local_offsets_of", []> {
    let summary = "Get local offsets of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs Variadic<Index>:$l_offsets);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        auto rank = d_tensor.getType().dyn_cast<::imex::dist::DistTensorType>().getPTensorType().getRank();
        auto IndexType = $_builder.getIndexType();
        ::mlir::SmallVector<::mlir::Type> rt(rank, IndexType);
        build($_builder, $_state, ::mlir::TypeRange(rt), d_tensor);
      }]>,
    ];
}

def LocalTensorOfOp : Dist_Op<"local_tensor_of", []> {
    let summary = "Get local tensor of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs AnyType:$l_tensor);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        auto ttype = d_tensor.getType().dyn_cast<::imex::dist::DistTensorType>();
        build($_builder, $_state, ttype.getPTensorType(), d_tensor);
      }]>,
    ];
}

def TeamOfOp : Dist_Op<"team_of", []> {
    let summary = "Get team of distributed tensor.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs AnyType:$team);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        build($_builder, $_state, $_builder.getIndexType(), d_tensor);
      }]>,
    ];
}

def IsBalancedOp : Dist_Op<"is_balanced", []> {
    let summary = "Determines whether the tensor is balanced or not.";
    let arguments = (ins AnyType:$d_tensor);
    let results = (outs Index:$is_balanced);
    let builders = [
      // auto-deduce return type from from operands
      OpBuilder<(ins "::mlir::Value":$d_tensor), [{
        build($_builder, $_state, $_builder.getIndexType(), d_tensor);
      }]>,
    ];
}

def LocalPartitionOp : Dist_Op<"local_partition", [SameVariadicResultSize, Pure]> {
    let summary = "Compute the shape and offsets of the local partition in number of elements (one for each dimension).";
    let arguments = (ins Index:$num_procs, Index:$p_rank, Variadic<Index>:$g_shape);
    let results = (outs Variadic<Index>:$l_offsets, Variadic<Index>:$l_shape);
    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$num_procs, "::mlir::Value":$prank, "::mlir::ValueRange":$gshape), [{
            auto IndexType = $_builder.getIndexType();
            ::mlir::SmallVector<::mlir::Type> rt(gshape.size()*2, IndexType);
            build($_builder,
                  $_state,
                  ::mlir::TypeRange(rt),
                  num_procs,
                  prank,
                  gshape);
        }]>,
    ];
}

def LocalOffsetForTargetSliceOp : Dist_Op<"local_offset_for_target_slice",
    [SameVariadicOperandSize, Pure]> {
    let summary = "Compute local offset of a given requested partition of a given slice";
    let description = [{
        Slice and tensor operate on the global index space. This operation computes the
        local part of the slice as provided as target - assuming it is owned locally. The operation
        returns local offsets relative to the local data/tensor.
    }];

    let arguments = (ins
        Variadic<Index>:$base_offsets,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$offsets,
        Variadic<Index>:$strides
    );
    let results = (outs Variadic<Index>:$l_offsets);

    let assemblyFormat = [{
        `[` $base_offsets `]``[` $target_offsets `]` `[` $offsets `]``[` $strides `]` attr-dict `->` `(`qualified(type(results))`)`
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::ValueRange":$boffsets, "::mlir::ValueRange":$toffsets, "::mlir::ValueRange":$offsets, "::mlir::ValueRange":$strides), [{
            auto IndexType = $_builder.getIndexType();
            ::mlir::SmallVector<::mlir::Type> rt(offsets.size(), IndexType);
            build($_builder, $_state, ::mlir::TypeRange(rt), boffsets, toffsets, offsets, strides);
        }]>,
    ];
}

def LocalTargetOfSliceOp : Dist_Op<"local_target_of_slice",
    [SameVariadicOperandSize, SameVariadicResultSize, Pure]> {
    let summary = "Compute local overlap of a distributed tensor and slice";
    let description = [{
        Slice and tensor operate on the global index space. This operation computes the
        local part of the slice as owned by the local partition of the tensor. The operation
        returns local offsets and sizes relative to the local tensor.
    }];

    let arguments = (ins
        Dist_Tensor:$d_tensor,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides
    );
    let results = (outs Variadic<Index>:$t_offsets, Variadic<Index>:$t_sizes);

    let assemblyFormat = [{
        $d_tensor `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($d_tensor)) `to` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$src, "::mlir::ValueRange":$offsets, "::mlir::ValueRange":$sizes, "::mlir::ValueRange":$strides), [{
            auto IndexType = $_builder.getIndexType();
            ::mlir::SmallVector<::mlir::Type> rt(offsets.size()*2, IndexType);
            build($_builder, $_state, ::mlir::TypeRange(rt), src, offsets, sizes, strides);
        }]>,
    ];
}

def LocalBoundingBoxOp : Dist_Op<"local_bounding_box", [AttrSizedOperandSegments, SameVariadicResultSize, Pure]> {
    let summary = "Compute bounding box for data locally needed by given views.";
    let description = [{
    }];

    let arguments = (ins Dist_Tensor:$base, Variadic<Index>:$offsets, Variadic<Index>:$sizes, Variadic<Index>:$strides,
                         Variadic<Index>:$target_offsets, Variadic<Index>:$target_sizes,
                         Variadic<Index>:$b_b_offsets, Variadic<Index>:$b_b_sizes);
    let results = (outs Variadic<Index>:$result_offsets, Variadic<Index>:$result_sizes);

    let assemblyFormat = [{
        $base `[` $offsets `]``[` $sizes `]``[` $strides `]` `[` $target_offsets `]``[` $target_sizes `]` oilist(`bboffs` $b_b_offsets | `bb_sizes` $b_b_sizes) attr-dict `:` qualified(type($base)) `to` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type: same as input
        OpBuilder<(ins "::mlir::Value":$base, "::mlir::ValueRange":$offs, "::mlir::ValueRange":$sizes, "::mlir::ValueRange":$strides,
                       "::mlir::ValueRange":$toffs, "::mlir::ValueRange":$tsizes,
                       "::mlir::ValueRange":$bboffs, "::mlir::ValueRange":$bbsizes), [{
            size_t rank = base.getType().dyn_cast<::imex::dist::DistTensorType>().getPTensorType().getRank();
            assert(offs.size() == rank && sizes.size() == rank);
            ::mlir::SmallVector<::mlir::Type> rt(2 * rank, $_builder.getIndexType());
            build($_builder, $_state, ::mlir::TypeRange(rt), base, offs, sizes, strides, toffs, tsizes, bboffs, bbsizes);
        }]>,
    ];
}

def RePartitionOp : Dist_Op<"repartition", [AttrSizedOperandSegments, Pure]> {
    let summary = "Repartition a tensor so that each process hold data locally needed by given views.";
    let description = [{
        Creates a new DistTensor by repartitioning the input tensor. It is assumed to be a collective call
        and participating processes report the target part they locally need.
    }];

    let arguments = (ins Dist_Tensor:$base, Variadic<Index>:$target_offsets, Variadic<Index>:$target_sizes);
    let results = (outs Dist_Tensor);

    let assemblyFormat = [{
        $base oilist(`loffs` $target_offsets | `lsizes` $target_sizes) attr-dict `:` qualified(type(operands)) `to` qualified(type(results))
    }];

    let builders = [
        // auto-deduce return type: same as input
        OpBuilder<(ins "::mlir::Value":$base), [{
            build($_builder, $_state, base.getType(), base, {}, {});
        }]>,
    ];
}

def LocalToGlobalOp : Dist_Op<"local_to_global", [Pure]> {
    let summary = "Translate local indices into global indices";
    let description = [{
        Input indices are interpreted as relative to the local part of the given DTensor.
    }];

    let arguments = (ins AnyType:$d_tensor, Variadic<Index>:$l_indices);
    let results = (outs Variadic<Index>:$g_indices);

    let builders = [
        // auto-deduce return type
        OpBuilder<(ins "::mlir::Value":$d_tensor, "::mlir::ValueRange":$lindices), [{
            auto IndexType = $_builder.getIndexType();
            ::mlir::SmallVector<::mlir::Type> rt(lindices.size(), IndexType);
            build($_builder, $_state, ::mlir::TypeRange(rt), d_tensor, lindices);
        }]>,
    ];
}

def AllReduceOp : Dist_Op<"allreduce", []> {
    let summary = "Inplace allreduce";
    let description = [{
        Result is the allreduced input tensor.
    }];
    // reduction operation and local tensor
    let arguments = (ins AnyAttr:$op, AnyMemRef:$data);
    let results = (outs AnyType);
}

// ============================================================================
// (Extended) operations from PTensor
// ============================================================================

def SubviewOp : Dist_Op<"subview", [AttrSizedOperandSegments, Pure]> {
    let summary = "distributed extract slice operation";
    let description = [{
        The "subview" operation extract a tensor from another tensor as
        specified by the operation's offsets, sizes and strides arguments.

        The distributed subview operation is a shallow wrapper around PTensor.subview
        and adds offsets and slices for targeted local partition.
    }];

    let arguments = (ins
        AnyType:$source,
        Variadic<Index>:$offsets,
        Variadic<Index>:$sizes,
        Variadic<Index>:$strides,
        DenseI64ArrayAttr:$static_offsets,
        DenseI64ArrayAttr:$static_sizes,
        DenseI64ArrayAttr:$static_strides,
        Variadic<Index>:$target_offsets,
        Variadic<Index>:$target_sizes
    );
    let results = (outs AnyType:$result);

    let assemblyFormat = [{
        $source ``
        custom<DynamicIndexList>($offsets, $static_offsets)
        custom<DynamicIndexList>($sizes, $static_sizes)
        custom<DynamicIndexList>($strides, $static_strides)
        attr-dict `:` qualified(type($source)) `to` qualified(type($result))
        oilist(`toffs` $target_offsets | `tsizes` $target_sizes)
    }];
}

def LoadOp : Dist_Op<"load",
     [AttrSizedOperandSegments, Pure,
      TypesMatchWith<"result type matches element type of 'array'",
                     "array", "result",
                     "$_self.cast<DistTensorType>().getPTensorType().getElementType()">]> {
  let summary = "array element load operation";
  let description = [{
    The `load` op reads an element from a array specified by an index list. The
    output of load is a new value with the same type as the elements of the
    array. The arity of indices is the rank of the array (i.e., if the array
    loaded from is of rank 3, then 3 indices are required for the load following
    the array identifier).
  }];

  let arguments = (ins AnyType:$array,
                       Variadic<Index>:$indices,
                       Variadic<Index>:$target_offsets);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "::mlir::Value":$array, CArg<"::mlir::ValueRange", "{}">:$indices), [{
      auto arrayType = array.getType().cast<DistTensorType>();
      build($_builder, $_state, arrayType.getElementType(), array, indices, {});
    }]>];

  let assemblyFormat = "$array `[` $indices `]` attr-dict `:` qualified(type($array)) oilist(`toffs` $target_offsets)";

  // let hasFolder = 1;
  // let hasCanonicalizer = 1;
}


def InsertSliceOp : Dist_Op<"insert_slice", [AttrSizedOperandSegments]> {
  let summary = "Copy values from a distributed tensor into a slice of another.";
  let description = [{
    Copy values from a distributed tensor into a slice of another.
  }];

  let arguments = (ins
    AnyType:$destination,
    AnyType:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    Variadic<Index>:$strides,
    Variadic<Index>:$target_offsets,
    Variadic<Index>:$target_sizes
  );

  let assemblyFormat = [{
    $source `into` $destination `[` $offsets `]``[` $sizes `]``[` $strides `]` attr-dict `:` qualified(type($destination)) `into` qualified(type($source)) oilist(`toffs` $target_offsets | `tsizes` $target_sizes)
  }];
}

#endif // _Dist_OPS_TD_INCLUDED_
