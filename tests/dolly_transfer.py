import numpy as np
import tvm
from tvm import meta_schedule as ms
from tvm.script import tir as T

from dolly_single import q_matmul5_32


def mutate_to_dyn_m(trace: tvm.tir.schedule.Trace, name_map) -> tvm.tir.schedule.Trace:
    rv_map = {}
    def map_to(arg):
        return rv_map[arg] if arg in rv_map else arg
    
    def map_attrs_to(attr):
        if isinstance(attr, tvm.runtime.container.String) and attr in name_map:
            return name_map[attr]
        return attr

    def just_copy(inst):
        return tvm.tir.schedule.Instruction(
            inst.kind,
            [map_to(inp) for inp in inst.inputs],
            [map_attrs_to(attr) for attr in inst.attrs],
            [map_to(outp) for outp in inst.outputs]
        )

    def process_SampleCategorical(inst: tvm.tir.schedule.Instruction):
        decision = int(trace.decisions[inst])
        val = inst.attrs[0][decision]
        rv_map[inst.outputs[0]] = val
        return []

    first_SamplePerfectTile = True 
    def process_SamplePerfectTile(inst: tvm.tir.schedule.Instruction):
        decision = [int(des) for des in trace.decisions[inst]]

        nonlocal first_SamplePerfectTile
        if first_SamplePerfectTile:
            first_SamplePerfectTile = False
            decision[0] = None
        
        for rv, val in zip(inst.outputs, decision):
            rv_map[rv] = T.int64(val) if val is not None else None

        return []

    rv_matmul, rv_p_in, rv_p_out = None, None, None
    def process_GetBlock(inst: tvm.tir.schedule.Instruction):
        nonlocal rv_matmul
        if rv_matmul is None and inst.attrs[0] == "matmul":
            rv_matmul = inst.outputs[0]
        
        return [just_copy(inst)]
    
    def process_GetLoops(inst: tvm.tir.schedule.Instruction):
        nonlocal rv_matmul, rv_p_in, rv_p_out
        if inst.inputs[0] == rv_matmul and len(inst.outputs) == 3:
            pad = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("PadEinsum"),
                [rv_matmul], [[T.int64(32), T.int64(16), T.int64(16)]], []
            )
            p_in = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("GetProducers"),
                [rv_matmul], [], [tvm.tir.schedule.BlockRV(), tvm.tir.schedule.BlockRV()]
            )            
            p_out = tvm.tir.schedule.Instruction(
                tvm.tir.schedule.InstructionKind.get("GetConsumers"),
                [rv_matmul], [], [tvm.tir.schedule.BlockRV()]
            )
            rv_p_in, rv_p_out = p_in.outputs[0], p_out.outputs[0]

            return [pad, p_in, p_out, just_copy(inst)]
        else:
            return [just_copy(inst)]
    
    def process_EnterPostproc(inst: tvm.tir.schedule.Instruction):
        inlibe_a = tvm.tir.schedule.Instruction(
            tvm.tir.schedule.InstructionKind.get("ComputeInline"),
            [rv_p_in], [], []
        )
        inlibe_b = tvm.tir.schedule.Instruction(
            tvm.tir.schedule.InstructionKind.get("ReverseComputeInline"),
            [rv_p_out], [], []
        )
        return [inlibe_a, inlibe_b, just_copy(inst)]

    processing_funcs ={
        "SamplePerfectTile": process_SamplePerfectTile,
        "SampleCategorical": process_SampleCategorical,
        "GetBlock": process_GetBlock,
        "GetLoops": process_GetLoops,
        "EnterPostproc": process_EnterPostproc,
    }

    new_insts = []
    for inst in trace.insts:
        if inst.kind.name in processing_funcs:
            for inst_ in processing_funcs[inst.kind.name](inst):
                new_insts.append(inst_)
        else:
            new_insts.append(just_copy(inst))

    return tvm.tir.schedule.Trace(new_insts, {})


@T.prim_func
def q_matmul2(var_A: T.handle, B: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), C: T.Buffer((T.int64(160), T.int64(640)), "int32"), D: T.Buffer((T.int64(160), T.int64(5120)), "float16"), E: T.Buffer((T.int64(5120),), "float16"), var_T_add: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(20480)), "float16")
    T_add = T.match_buffer(var_T_add, (T.int64(1), n, T.int64(5120)), "float16")
    # with T.block("root"):
    reshape_in = T.alloc_buffer((n, T.int64(20480)), "float16")
    decode_zp = T.alloc_buffer((T.int64(160), T.int64(5120)), "int32")
    decode_wgh = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16")
    matmul = T.alloc_buffer((n, T.int64(5120)), "float16")
    reshape_out = T.alloc_buffer((T.int64(1), n, T.int64(5120)), "float16")
    for vm, vk in T.grid(n, T.int64(20480)):
        with T.block("reshape_in"):
            v_vm, v_vk = T.axis.remap("SS", [vm, vk])
            T.reads(A[T.int64(0), v_vm, v_vk])
            T.writes(reshape_in[v_vm, v_vk])
            reshape_in[v_vm, v_vk] = A[T.int64(0), v_vm, v_vk]
    for vg, vn in T.grid(T.int64(160), T.int64(5120)):
        with T.block("decode_zp"):
            v_vg, v_vn = T.axis.remap("SS", [vg, vn])
            T.reads(C[v_vg, v_vn // T.int64(8)])
            T.writes(decode_zp[v_vg, v_vn])
            decode_zp[v_vg, v_vn] = T.bitwise_and(T.shift_right(C[v_vg, v_vn // T.int64(8)], T.Cast("int32", v_vn) % 8 * 4), 15) + 1
    for vk, vn in T.grid(T.int64(20480), T.int64(5120)):
        with T.block("decode_wgh"):
            v_vk, v_vn = T.axis.remap("SS", [vk, vn])
            T.reads(B[v_vk // T.int64(8), v_vn], decode_zp[v_vk // T.int64(128), v_vn], D[v_vk // T.int64(128), v_vn])
            T.writes(decode_wgh[v_vk, v_vn])
            decode_wgh[v_vk, v_vn] = T.Cast("float16", T.bitwise_and(T.shift_right(B[v_vk // T.int64(8), v_vn], T.Cast("int32", v_vk) % 8 * 4), 15) - decode_zp[v_vk // T.int64(128), v_vn]) * D[v_vk // T.int64(128), v_vn]
    for vm, vn, k in T.grid(n, T.int64(5120), T.int64(20480)):
        with T.block("matmul"):
            v_vm, v_vn, v_k = T.axis.remap("SSR", [vm, vn, k])
            T.reads(reshape_in[v_vm, v_k], decode_wgh[v_k, v_vn])
            T.writes(matmul[v_vm, v_vn])
            with T.init():
                matmul[v_vm, v_vn] = T.float16(0)
            matmul[v_vm, v_vn] = matmul[v_vm, v_vn] + reshape_in[v_vm, v_k] * decode_wgh[v_k, v_vn]
    for vb, vm, vn in T.grid(T.int64(1), n, T.int64(5120)):
        with T.block("reshape_out"):
            v_vb, v_vm, v_vn = T.axis.remap("SSS", [vb, vm, vn])
            T.reads(matmul[v_vm, v_vn])
            T.writes(reshape_out[v_vb, v_vm, v_vn])
            reshape_out[v_vb, v_vm, v_vn] = matmul[v_vm, v_vn]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(reshape_out[v_ax0, v_ax1, v_ax2], E[v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = reshape_out[v_ax0, v_ax1, v_ax2] + E[v_ax2]

def apply_s_trace(sch: tvm.tir.Schedule) -> None:
  b0 = sch.get_block(name="reshape_in", func_name="main")
  b1 = sch.get_block(name="decode_zp", func_name="main")
  b2 = sch.get_block(name="decode_wgh", func_name="main")
  b3 = sch.get_block(name="matmul", func_name="main")
  b4 = sch.get_block(name="reshape_out", func_name="main")
  b5 = sch.get_block(name="T_add", func_name="main")
  b6 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  b7 = sch.reindex(block=b3, buffer=("write", 0))
  b8 = sch.reindex(block=b3, buffer=("read", 0))
  b9 = sch.reindex(block=b3, buffer=("read", 1))
  sch.transform_layout(block=b3, buffer=("read", 0), index_map=lambda v_vm, v_k: (v_vm, v_k,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("read", 1), index_map=lambda v_vn, v_k: (v_k, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("write", 0), index_map=lambda v_vm, v_vn: (v_vm, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_block_layout(block=b7, index_map=lambda v_vm, v_vn: (v_vm, v_vn,))
  sch.transform_block_layout(block=b8, index_map=lambda v_vm, v_k: (v_vm, v_k,))
  sch.transform_block_layout(block=b9, index_map=lambda v_vn, v_k: (v_k, v_vn,))
  sch.transform_block_layout(block=b3, index_map=lambda v_vm, v_vn, v_k: (v_vm, v_vn, v_k,))
  l10, l11, l12 = sch.get_loops(block=b3)
  l13, l14 = sch.split(loop=l12, factors=[None, 16], preserve_unit_iters=True)
  l15, l16 = sch.split(loop=l11, factors=[None, 16], preserve_unit_iters=True)
  l17, l18 = sch.split(loop=l10, factors=[None, 16], preserve_unit_iters=True)
  l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b3)
  sch.reorder(l21, l23, l18, l16, l14)
  b25 = sch.blockize(target=l18, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
  sch.annotate(block_or_loop=b25, ann_key="warp_execution", ann_val=1)
  l26, l27, l28 = sch.get_loops(block=b25)
  v29, v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l26, n=5, max_innermost_factor=4, decision=[1, 1, 2, 1, 1])
  l34, l35, l36, l37, l38 = sch.split(loop=l26, factors=[v29, v30, v31, v32, v33], preserve_unit_iters=True)
  v39, v40, v41, v42, v43 = sch.sample_perfect_tile(loop=l27, n=5, max_innermost_factor=4, decision=[10, 32, 1, 1, 1])
  l44, l45, l46, l47, l48 = sch.split(loop=l27, factors=[v39, v40, v41, v42, v43], preserve_unit_iters=True)
  v49, v50, v51 = sch.sample_perfect_tile(loop=l28, n=3, max_innermost_factor=4, decision=[160, 8, 1])
  l52, l53, l54 = sch.split(loop=l28, factors=[v49, v50, v51], preserve_unit_iters=True)
  sch.reorder(l34, l44, l35, l45, l36, l46, l52, l53, l37, l47, l54, l38, l48)
  l55 = sch.fuse(l34, l44, preserve_unit_iters=True)
  sch.bind(loop=l55, thread_axis="blockIdx.y")
  l56 = sch.fuse(l35, l45, preserve_unit_iters=True)
  sch.bind(loop=l56, thread_axis="blockIdx.x")
  l57 = sch.fuse(l36, l46, preserve_unit_iters=True)
  sch.bind(loop=l57, thread_axis="threadIdx.y")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  b58 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="shared.dyn")
  sch.reverse_compute_at(block=b58, loop=l56, preserve_unit_loops=True, index=-1)
  b59 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="wmma.accumulator")
  sch.reverse_compute_at(block=b59, loop=l57, preserve_unit_loops=True, index=-1)
  l60, l61, l62, l63 = sch.get_loops(block=b58)
  l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
  v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
  sch.reverse_compute_inline(block=b7)
  l66, l67, l68, l69, l70 = sch.get_loops(block=b59)
  l71, l72 = sch.split(loop=l70, factors=[None, 16], preserve_unit_iters=True)
  l73, l74 = sch.split(loop=l69, factors=[None, 16], preserve_unit_iters=True)
  l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b59)
  sch.reorder(l80, l74, l72)
  b82 = sch.blockize(target=l74, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b82, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
  b83 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b83, loop=l52, preserve_unit_loops=True, index=-1)
  l84, l85, l86, l87, l88, l89 = sch.get_loops(block=b83)
  l90 = sch.fuse(l88, l89, preserve_unit_iters=True)
  v91 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b83, ann_key="meta_schedule.cooperative_fetch", ann_val=v91)
  b92 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b92, loop=l52, preserve_unit_loops=True, index=-1)
  l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b92)
  l99 = sch.fuse(l97, l98, preserve_unit_iters=True)
  v100 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch", ann_val=v100)
  b101 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="wmma.matrix_a")
  sch.compute_at(block=b101, loop=l53, preserve_unit_loops=True, index=-1)
  l102, l103, l104, l105, l106, l107, l108 = sch.get_loops(block=b101)
  l109, l110 = sch.split(loop=l108, factors=[None, 16], preserve_unit_iters=True)
  l111, l112 = sch.split(loop=l107, factors=[None, 16], preserve_unit_iters=True)
  l113, l114, l115, l116, l117, l118, l119, l120, l121 = sch.get_loops(block=b101)
  sch.reorder(l120, l112, l110)
  b122 = sch.blockize(target=l112, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b122, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
  b123 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="wmma.matrix_b")
  sch.compute_at(block=b123, loop=l53, preserve_unit_loops=True, index=-1)
  l124, l125, l126, l127, l128, l129, l130 = sch.get_loops(block=b123)
  l131, l132 = sch.split(loop=l130, factors=[None, 16], preserve_unit_iters=True)
  l133, l134 = sch.split(loop=l129, factors=[None, 16], preserve_unit_iters=True)
  l135, l136, l137, l138, l139, l140, l141, l142, l143 = sch.get_loops(block=b123)
  sch.reorder(l142, l134, l132)
  b144 = sch.blockize(target=l134, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b144, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
  sch.compute_inline(block=b8)
  sch.compute_inline(block=b9)
  sch.storage_align(block=b83, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b92, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.reverse_compute_inline(block=b5)
  sch.reverse_compute_inline(block=b4)
  sch.compute_inline(block=b2)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  v145 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
  sch.annotate(block_or_loop=b6, ann_key="meta_schedule.unroll_explicit", ann_val=v145)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
  l146, l147, l148 = sch.get_loops(block=b58)
  l149, l150, l151, l152 = sch.split(loop=l148, factors=[None, 2, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l152)
  sch.bind(loop=l151, thread_axis="threadIdx.x")
  sch.bind(loop=l150, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.cooperative_fetch")
  l153, l154, l155, l156, l157 = sch.get_loops(block=b83)
  l158, l159, l160, l161 = sch.split(loop=l157, factors=[None, 2, 32, 8], preserve_unit_iters=True)
  sch.vectorize(loop=l161)
  sch.bind(loop=l160, thread_axis="threadIdx.x")
  sch.bind(loop=l159, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch")
  l162, l163, l164, l165, l166 = sch.get_loops(block=b92)
  l167, l168, l169 = sch.split(loop=l166, factors=[None, 2, 32], preserve_unit_iters=True)
  sch.bind(loop=l169, thread_axis="threadIdx.x")
  sch.bind(loop=l168, thread_axis="threadIdx.y")
  b170 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b170, ann_key="meta_schedule.unroll_explicit")
  b171, b172, b173, b174, b175, b176, b177 = sch.get_child_blocks(b170)
  l178, l179, l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b171)
  l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b172)
  l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b173)
  l200, l201, l202, l203, l204, l205, l206 = sch.get_loops(block=b174)
  l207, l208, l209, l210, l211, l212, l213, l214, l215, l216 = sch.get_loops(block=b175)
  sch.annotate(block_or_loop=l207, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l207, ann_key="pragma_unroll_explicit", ann_val=1)
  l217, l218, l219, l220, l221 = sch.get_loops(block=b176)
  l222, l223, l224, l225, l226, l227 = sch.get_loops(block=b177)
  b228 = sch.get_block(name="matmul_o", func_name="main")
  l229, l230, l231, l232, l233, l234, l235, l236, l237, l238 = sch.get_loops(block=b228)
  b239 = sch.decompose_reduction(block=b228, loop=l232)
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b228, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize_init")
  b240 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b240, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b240, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
  b241 = sch.get_block(name="reshape_in_reindex_shared.dyn_wmma.matrix_a_o", func_name="main")
  sch.unannotate(block_or_loop=b241, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b241, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b242 = sch.get_block(name="decode_wgh_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b242, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b242, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b243 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b243, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b243, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
  b244 = sch.get_block(name="matmul_reindex_shared.dyn_wmma.accumulator_o", func_name="main")
  sch.unannotate(block_or_loop=b244, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b244, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


def apply_d_trace(sch: tvm.tir.Schedule) -> None:
  b0 = sch.get_block(name="reshape_in", func_name="main")
  b1 = sch.get_block(name="decode_zp", func_name="main")
  b2 = sch.get_block(name="decode_wgh", func_name="main")
  b3 = sch.get_block(name="matmul", func_name="main")
  b4 = sch.get_block(name="reshape_out", func_name="main")
  b5 = sch.get_block(name="T_add", func_name="main")
  b6 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  b7 = sch.reindex(block=b3, buffer=("write", 0))
  b8 = sch.reindex(block=b3, buffer=("read", 0))
  b9 = sch.reindex(block=b3, buffer=("read", 1))
  sch.transform_layout(block=b3, buffer=("read", 0), index_map=lambda v_vm, v_k: (v_vm, v_k,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("read", 1), index_map=lambda v_vn, v_k: (v_k, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("write", 0), index_map=lambda v_vm, v_vn: (v_vm, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_block_layout(block=b7, index_map=lambda v_vm, v_vn: (v_vm, v_vn,))
  sch.transform_block_layout(block=b8, index_map=lambda v_vm, v_k: (v_vm, v_k,))
  sch.transform_block_layout(block=b9, index_map=lambda v_vn, v_k: (v_k, v_vn,))
  sch.transform_block_layout(block=b3, index_map=lambda v_vm, v_vn, v_k: (v_vm, v_vn, v_k,))
  
  sch.pad_einsum(b3, [32, 16, 16])  # [XXX]
  p_in, _ = sch.get_producers(b3) 
  p_out, = sch.get_consumers(b3) 

  l10, l11, l12 = sch.get_loops(block=b3)
  l13, l14 = sch.split(loop=l12, factors=[None, 16], preserve_unit_iters=True)
  l15, l16 = sch.split(loop=l11, factors=[None, 16], preserve_unit_iters=True)
  l17, l18 = sch.split(loop=l10, factors=[None, 16], preserve_unit_iters=True)
  l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b3)
  sch.reorder(l21, l23, l18, l16, l14)
  b25 = sch.blockize(target=l18, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
  sch.annotate(block_or_loop=b25, ann_key="warp_execution", ann_val=1)
  l26, l27, l28 = sch.get_loops(block=b25)
#   v29, v30, v31, v32, v33 = sch.sample_perfect_tile(loop=l26, n=5, max_innermost_factor=4, decision=[1, 1, 2, 1, 1])
#   l34, l35, l36, l37, l38 = sch.split(loop=l26, factors=[v29, v30, v31, v32, v33], preserve_unit_iters=True)
  l34, l35, l36, l37, l38 = sch.split(loop=l26, factors=[None, 1, 2, 1, 1], preserve_unit_iters=True)   # [XXX]
  v39, v40, v41, v42, v43 = sch.sample_perfect_tile(loop=l27, n=5, max_innermost_factor=4, decision=[10, 32, 1, 1, 1])
  l44, l45, l46, l47, l48 = sch.split(loop=l27, factors=[v39, v40, v41, v42, v43], preserve_unit_iters=True)
  v49, v50, v51 = sch.sample_perfect_tile(loop=l28, n=3, max_innermost_factor=4, decision=[160, 8, 1])
  l52, l53, l54 = sch.split(loop=l28, factors=[v49, v50, v51], preserve_unit_iters=True)
  sch.reorder(l34, l44, l35, l45, l36, l46, l52, l53, l37, l47, l54, l38, l48)
  l55 = sch.fuse(l34, l44, preserve_unit_iters=True)
  sch.bind(loop=l55, thread_axis="blockIdx.y")
  l56 = sch.fuse(l35, l45, preserve_unit_iters=True)
  sch.bind(loop=l56, thread_axis="blockIdx.x")
  l57 = sch.fuse(l36, l46, preserve_unit_iters=True)
  sch.bind(loop=l57, thread_axis="threadIdx.y")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  b58 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="shared.dyn")
  sch.reverse_compute_at(block=b58, loop=l56, preserve_unit_loops=True, index=-1)
  b59 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="wmma.accumulator")
  sch.reverse_compute_at(block=b59, loop=l57, preserve_unit_loops=True, index=-1)
  l60, l61, l62, l63 = sch.get_loops(block=b58)
  l64 = sch.fuse(l62, l63, preserve_unit_iters=True)
  v65 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=1)
  sch.annotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch", ann_val=v65)
  sch.reverse_compute_inline(block=b7)
  l66, l67, l68, l69, l70 = sch.get_loops(block=b59)
  l71, l72 = sch.split(loop=l70, factors=[None, 16], preserve_unit_iters=True)
  l73, l74 = sch.split(loop=l69, factors=[None, 16], preserve_unit_iters=True)
  l75, l76, l77, l78, l79, l80, l81 = sch.get_loops(block=b59)
  sch.reorder(l80, l74, l72)
  b82 = sch.blockize(target=l74, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b82, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
  b83 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b83, loop=l52, preserve_unit_loops=True, index=-1)
  l84, l85, l86, l87, l88, l89 = sch.get_loops(block=b83)
  l90 = sch.fuse(l88, l89, preserve_unit_iters=True)
  v91 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=3)
  sch.annotate(block_or_loop=b83, ann_key="meta_schedule.cooperative_fetch", ann_val=v91)
  b92 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b92, loop=l52, preserve_unit_loops=True, index=-1)
  l93, l94, l95, l96, l97, l98 = sch.get_loops(block=b92)
  l99 = sch.fuse(l97, l98, preserve_unit_iters=True)
  v100 = sch.sample_categorical(candidates=[1, 2, 4, 8], probs=[0.25, 0.25, 0.25, 0.25], decision=0)
  sch.annotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch", ann_val=v100)
  b101 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="wmma.matrix_a")
  sch.compute_at(block=b101, loop=l53, preserve_unit_loops=True, index=-1)
  l102, l103, l104, l105, l106, l107, l108 = sch.get_loops(block=b101)
  l109, l110 = sch.split(loop=l108, factors=[None, 16], preserve_unit_iters=True)
  l111, l112 = sch.split(loop=l107, factors=[None, 16], preserve_unit_iters=True)
  l113, l114, l115, l116, l117, l118, l119, l120, l121 = sch.get_loops(block=b101)
  sch.reorder(l120, l112, l110)
  b122 = sch.blockize(target=l112, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b122, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
  b123 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="wmma.matrix_b")
  sch.compute_at(block=b123, loop=l53, preserve_unit_loops=True, index=-1)
  l124, l125, l126, l127, l128, l129, l130 = sch.get_loops(block=b123)
  l131, l132 = sch.split(loop=l130, factors=[None, 16], preserve_unit_iters=True)
  l133, l134 = sch.split(loop=l129, factors=[None, 16], preserve_unit_iters=True)
  l135, l136, l137, l138, l139, l140, l141, l142, l143 = sch.get_loops(block=b123)
  sch.reorder(l142, l134, l132)
  b144 = sch.blockize(target=l134, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b144, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")
  sch.compute_inline(block=b8)
  sch.compute_inline(block=b9)

#   sch.compute_inline("reshape_in_reindex_pad")  # [XXX]
  sch.compute_inline(p_in)  # [XXX]

  sch.storage_align(block=b83, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b92, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.reverse_compute_inline(block=b5)
  sch.reverse_compute_inline(block=b4)
  
#   sch.reverse_compute_inline("matmul_reindex_pad")  # [XXX]
  sch.reverse_compute_inline(p_out)  # [XXX]
  
  sch.compute_inline(block=b2)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  v145 = sch.sample_categorical(candidates=[0, 16, 64, 512, 1024], probs=[0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001], decision=1)
  sch.annotate(block_or_loop=b6, ann_key="meta_schedule.unroll_explicit", ann_val=v145)
  sch.enter_postproc()
  sch.unannotate(block_or_loop=b58, ann_key="meta_schedule.cooperative_fetch")
  l146, l147, l148 = sch.get_loops(block=b58)
  l149, l150, l151, l152 = sch.split(loop=l148, factors=[None, 2, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l152)
  sch.bind(loop=l151, thread_axis="threadIdx.x")
  sch.bind(loop=l150, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b83, ann_key="meta_schedule.cooperative_fetch")
  l153, l154, l155, l156, l157 = sch.get_loops(block=b83)
  l158, l159, l160, l161 = sch.split(loop=l157, factors=[None, 2, 32, 8], preserve_unit_iters=True)
  sch.vectorize(loop=l161)
  sch.bind(loop=l160, thread_axis="threadIdx.x")
  sch.bind(loop=l159, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b92, ann_key="meta_schedule.cooperative_fetch")
  l162, l163, l164, l165, l166 = sch.get_loops(block=b92)
  l167, l168, l169 = sch.split(loop=l166, factors=[None, 2, 32], preserve_unit_iters=True)
  sch.bind(loop=l169, thread_axis="threadIdx.x")
  sch.bind(loop=l168, thread_axis="threadIdx.y")
  b170 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b170, ann_key="meta_schedule.unroll_explicit")
  b171, b172, b173, b174, b175, b176, b177 = sch.get_child_blocks(b170)
  l178, l179, l180, l181, l182, l183, l184, l185 = sch.get_loops(block=b171)
  l186, l187, l188, l189, l190, l191, l192 = sch.get_loops(block=b172)
  l193, l194, l195, l196, l197, l198, l199 = sch.get_loops(block=b173)
  l200, l201, l202, l203, l204, l205, l206 = sch.get_loops(block=b174)
  l207, l208, l209, l210, l211, l212, l213, l214, l215, l216 = sch.get_loops(block=b175)
  sch.annotate(block_or_loop=l207, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l207, ann_key="pragma_unroll_explicit", ann_val=1)
  l217, l218, l219, l220, l221 = sch.get_loops(block=b176)
  l222, l223, l224, l225, l226, l227 = sch.get_loops(block=b177)
  b228 = sch.get_block(name="matmul_o", func_name="main")
  l229, l230, l231, l232, l233, l234, l235, l236, l237, l238 = sch.get_loops(block=b228)
  b239 = sch.decompose_reduction(block=b228, loop=l232)
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b228, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b239, ann_key="meta_schedule.auto_tensorize_init")
  b240 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b240, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b240, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
#   b241 = sch.get_block(name="reshape_in_reindex_shared.dyn_wmma.matrix_a_o", func_name="main")
  b241 = sch.get_block(name="reshape_in_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")  # [XXX]
  sch.unannotate(block_or_loop=b241, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b241, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b242 = sch.get_block(name="decode_wgh_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b242, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b242, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b243 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b243, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b243, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
#   b244 = sch.get_block(name="matmul_reindex_shared.dyn_wmma.accumulator_o", func_name="main")
  b244 = sch.get_block(name="matmul_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")  # [XXX]
  sch.unannotate(block_or_loop=b244, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b244, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)


# from tvm import tir
def apply_gen_trace(sch: tvm.tir.Schedule) -> None:
  b0 = sch.get_block(name="reshape_in", func_name="main")
  b1 = sch.get_block(name="decode_zp", func_name="main")
  b2 = sch.get_block(name="decode_wgh", func_name="main")
  b3 = sch.get_block(name="matmul", func_name="main")
  b4 = sch.get_block(name="reshape_out", func_name="main")
  b5 = sch.get_block(name="T_add", func_name="main")
  b6 = sch.get_block(name="root", func_name="main")
  sch.annotate(block_or_loop=b3, ann_key="meta_schedule.tiling_structure", ann_val="SSSRRSRS")
  b7 = sch.reindex(block=b3, buffer=("write", 0))
  b8 = sch.reindex(block=b3, buffer=("read", 0))
  b9 = sch.reindex(block=b3, buffer=("read", 1))
  sch.transform_layout(block=b3, buffer=("read", 0), index_map=lambda v_vm, v_k: (v_vm, v_k,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("read", 1), index_map=lambda v_vn, v_k: (v_k, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_layout(block=b3, buffer=("write", 0), index_map=lambda v_vm, v_vn: (v_vm, v_vn,), pad_value=None, assume_injective_transform=False)
  sch.transform_block_layout(block=b7, index_map=lambda v_vm, v_vn: (v_vm, v_vn,))
  sch.transform_block_layout(block=b8, index_map=lambda v_vm, v_k: (v_vm, v_k,))
  sch.transform_block_layout(block=b9, index_map=lambda v_vn, v_k: (v_k, v_vn,))
  sch.transform_block_layout(block=b3, index_map=lambda v_vm, v_vn, v_k: (v_vm, v_vn, v_k,))
  sch.pad_einsum(block=b3, padding=[32, 16, 16])
  l10, l11, l12 = sch.get_loops(block=b3)
  l13, l14 = sch.split(loop=l12, factors=[None, 16], preserve_unit_iters=True)
  l15, l16 = sch.split(loop=l11, factors=[None, 16], preserve_unit_iters=True)
  l17, l18 = sch.split(loop=l10, factors=[None, 16], preserve_unit_iters=True)
  l19, l20, l21, l22, l23, l24 = sch.get_loops(block=b3)
  sch.reorder(l21, l23, l18, l16, l14)
  b25 = sch.blockize(target=l18, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_sync_16x16x16_f16f16f16")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.auto_tensorize_init", ann_val="wmma_fill_16x16x16_f16")
  sch.annotate(block_or_loop=b25, ann_key="warp_execution", ann_val=1)
  l26, l27, l28 = sch.get_loops(block=b25)
  l29, l30, l31, l32, l33 = sch.split(loop=l26, factors=[None, 1, 2, 1, 1], preserve_unit_iters=True)
  l34, l35, l36, l37, l38 = sch.split(loop=l27, factors=[10, 32, 1, 1, 1], preserve_unit_iters=True)
  l39, l40, l41 = sch.split(loop=l28, factors=[160, 8, 1], preserve_unit_iters=True)
  sch.reorder(l29, l34, l30, l35, l31, l36, l39, l40, l32, l37, l41, l33, l38)
  l42 = sch.fuse(l29, l34, preserve_unit_iters=True)
  sch.bind(loop=l42, thread_axis="blockIdx.y")
  l43 = sch.fuse(l30, l35, preserve_unit_iters=True)
  sch.bind(loop=l43, thread_axis="blockIdx.x")
  l44 = sch.fuse(l31, l36, preserve_unit_iters=True)
  sch.bind(loop=l44, thread_axis="threadIdx.y")
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_low_inclusive", ann_val=32)
  sch.annotate(block_or_loop=b25, ann_key="meta_schedule.thread_extent_high_inclusive", ann_val=1024)
  b45 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="shared.dyn")
  sch.reverse_compute_at(block=b45, loop=l43, preserve_unit_loops=True, index=-1)
  b46 = sch.cache_write(block=b25, write_buffer_index=0, storage_scope="wmma.accumulator")
  sch.reverse_compute_at(block=b46, loop=l44, preserve_unit_loops=True, index=-1)
  l47, l48, l49, l50 = sch.get_loops(block=b45)
  l51 = sch.fuse(l49, l50, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b45, ann_key="meta_schedule.cooperative_fetch", ann_val=2)
  sch.reverse_compute_inline(block=b7)
  l52, l53, l54, l55, l56 = sch.get_loops(block=b46)
  l57, l58 = sch.split(loop=l56, factors=[None, 16], preserve_unit_iters=True)
  l59, l60 = sch.split(loop=l55, factors=[None, 16], preserve_unit_iters=True)
  l61, l62, l63, l64, l65, l66, l67 = sch.get_loops(block=b46)
  sch.reorder(l66, l60, l58)
  b68 = sch.blockize(target=l60, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b68, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_store_16x16x16_f16_shared_dyn")
  b69 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b69, loop=l39, preserve_unit_loops=True, index=-1)
  l70, l71, l72, l73, l74, l75 = sch.get_loops(block=b69)
  l76 = sch.fuse(l74, l75, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b69, ann_key="meta_schedule.cooperative_fetch", ann_val=8)
  b77 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="shared.dyn", consumer_blocks=[b25])
  sch.compute_at(block=b77, loop=l39, preserve_unit_loops=True, index=-1)
  l78, l79, l80, l81, l82, l83 = sch.get_loops(block=b77)
  l84 = sch.fuse(l82, l83, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b77, ann_key="meta_schedule.cooperative_fetch", ann_val=1)
  b85 = sch.cache_read(block=b25, read_buffer_index=0, storage_scope="wmma.matrix_a")
  sch.compute_at(block=b85, loop=l40, preserve_unit_loops=True, index=-1)
  l86, l87, l88, l89, l90, l91, l92 = sch.get_loops(block=b85)
  l93, l94 = sch.split(loop=l92, factors=[None, 16], preserve_unit_iters=True)
  l95, l96 = sch.split(loop=l91, factors=[None, 16], preserve_unit_iters=True)
  l97, l98, l99, l100, l101, l102, l103, l104, l105 = sch.get_loops(block=b85)
  sch.reorder(l104, l96, l94)
  b106 = sch.blockize(target=l96, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b106, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_a_shared_dyn")
  b107 = sch.cache_read(block=b25, read_buffer_index=1, storage_scope="wmma.matrix_b")
  sch.compute_at(block=b107, loop=l40, preserve_unit_loops=True, index=-1)
  l108, l109, l110, l111, l112, l113, l114 = sch.get_loops(block=b107)
  l115, l116 = sch.split(loop=l114, factors=[None, 16], preserve_unit_iters=True)
  l117, l118 = sch.split(loop=l113, factors=[None, 16], preserve_unit_iters=True)
  l119, l120, l121, l122, l123, l124, l125, l126, l127 = sch.get_loops(block=b107)
  sch.reorder(l126, l118, l116)
  b128 = sch.blockize(target=l118, preserve_unit_iters=True)
  sch.annotate(block_or_loop=b128, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_load_16x16x16_f16_b_shared_dyn")


  sch.compute_inline(block=b8)
  sch.compute_inline(block=b9)
  sch.storage_align(block=b69, buffer_index=0, axis=-2, factor=32, offset=8)
  sch.storage_align(block=b77, buffer_index=0, axis=-2, factor=32, offset=8)

  sch.reverse_compute_inline(block=b5)
  sch.reverse_compute_inline(block=b4)
  sch.compute_inline(block=b2)
  sch.compute_inline(block=b1)
  sch.compute_inline(block=b0)
  sch.annotate(block_or_loop=b6, ann_key="meta_schedule.unroll_explicit", ann_val=16)

  b129 = sch.get_block(name="reshape_in_reindex_pad", func_name="main")
  sch.compute_inline(block=b129)
  b130 = sch.get_block(name="matmul_reindex_pad", func_name="main")
  sch.reverse_compute_inline(block=b130)

  sch.enter_postproc()
  sch.unannotate(block_or_loop=b45, ann_key="meta_schedule.cooperative_fetch")
  l131, l132, l133 = sch.get_loops(block=b45)
  l134, l135, l136, l137 = sch.split(loop=l133, factors=[None, 2, 32, 2], preserve_unit_iters=True)
  sch.vectorize(loop=l137)
  sch.bind(loop=l136, thread_axis="threadIdx.x")
  sch.bind(loop=l135, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b69, ann_key="meta_schedule.cooperative_fetch")
  l138, l139, l140, l141, l142 = sch.get_loops(block=b69)
  l143, l144, l145, l146 = sch.split(loop=l142, factors=[None, 2, 32, 8], preserve_unit_iters=True)
  sch.vectorize(loop=l146)
  sch.bind(loop=l145, thread_axis="threadIdx.x")
  sch.bind(loop=l144, thread_axis="threadIdx.y")
  sch.unannotate(block_or_loop=b77, ann_key="meta_schedule.cooperative_fetch")
  l147, l148, l149, l150, l151 = sch.get_loops(block=b77)
  l152, l153, l154 = sch.split(loop=l151, factors=[None, 2, 32], preserve_unit_iters=True)
  sch.bind(loop=l154, thread_axis="threadIdx.x")
  sch.bind(loop=l153, thread_axis="threadIdx.y")
  b155 = sch.get_block(name="root", func_name="main")
  sch.unannotate(block_or_loop=b155, ann_key="meta_schedule.unroll_explicit")
  b156, b157, b158, b159, b160, b161, b162 = sch.get_child_blocks(b155)
  l163, l164, l165, l166, l167, l168, l169, l170 = sch.get_loops(block=b156)
  l171, l172, l173, l174, l175, l176, l177 = sch.get_loops(block=b157)
  l178, l179, l180, l181, l182, l183, l184 = sch.get_loops(block=b158)
  l185, l186, l187, l188, l189, l190, l191 = sch.get_loops(block=b159)
  l192, l193, l194, l195, l196, l197, l198, l199, l200, l201 = sch.get_loops(block=b160)
  sch.annotate(block_or_loop=l192, ann_key="pragma_auto_unroll_max_step", ann_val=16)
  sch.annotate(block_or_loop=l192, ann_key="pragma_unroll_explicit", ann_val=1)
  l202, l203, l204, l205, l206 = sch.get_loops(block=b161)
  l207, l208, l209, l210, l211, l212 = sch.get_loops(block=b162)
  b213 = sch.get_block(name="matmul_o", func_name="main")
  l214, l215, l216, l217, l218, l219, l220, l221, l222, l223 = sch.get_loops(block=b213)
  b224 = sch.decompose_reduction(block=b213, loop=l217)
  sch.unannotate(block_or_loop=b224, ann_key="meta_schedule.auto_tensorize")
  sch.annotate(block_or_loop=b224, ann_key="meta_schedule.auto_tensorize", ann_val="wmma_fill_16x16x16_f16")
  sch.unannotate(block_or_loop=b213, ann_key="meta_schedule.auto_tensorize_init")
  sch.unannotate(block_or_loop=b224, ann_key="meta_schedule.auto_tensorize_init")
  b225 = sch.get_block(name="matmul_o_init", func_name="main")
  sch.unannotate(block_or_loop=b225, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b225, tensor_intrin="wmma_fill_16x16x16_f16", preserve_unit_iters=True)
  b226 = sch.get_block(name="reshape_in_reindex_pad_shared.dyn_wmma.matrix_a_o", func_name="main")
  sch.unannotate(block_or_loop=b226, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b226, tensor_intrin="wmma_load_16x16x16_f16_a_shared_dyn", preserve_unit_iters=True)
  b227 = sch.get_block(name="decode_wgh_reindex_shared.dyn_wmma.matrix_b_o", func_name="main")
  sch.unannotate(block_or_loop=b227, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b227, tensor_intrin="wmma_load_16x16x16_f16_b_shared_dyn", preserve_unit_iters=True)
  b228 = sch.get_block(name="matmul_o_update", func_name="main")
  sch.unannotate(block_or_loop=b228, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b228, tensor_intrin="wmma_sync_16x16x16_f16f16f16", preserve_unit_iters=True)
  b229 = sch.get_block(name="matmul_reindex_pad_shared.dyn_wmma.accumulator_o", func_name="main")
  sch.unannotate(block_or_loop=b229, ann_key="meta_schedule.auto_tensorize")
  sch.tensorize(block_or_loop=b229, tensor_intrin="wmma_store_16x16x16_f16_shared_dyn", preserve_unit_iters=True)

def generate_arg(info: ms.arg_info.ArgInfo, dev):
    if info.dtype == "float16":
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype('float16')
    elif info.dtype == "int32":
        arr_np = np.random.randint(0, 16, size=info.shape).astype('int32')
    else:
        assert False, "Unimplemented"

    return tvm.nd.array(arr_np, device=dev)


def main():
    db_name = "dolly_tune_tir_4b"
    dyn_db_name = "dolly_tune_tir_4b_dyn"

    dev = tvm.cuda(0)
    target = tvm.target.Target("nvidia/nvidia-a10g")
    top_k_to_analize = 40

    s_mod = q_matmul5_32
    d_mod = q_matmul2

    db = ms.database.JSONDatabase(work_dir=db_name, allow_missing=False)
    s_mod_norm = ms.tir_integration._normalize_mod(s_mod)
    rec = db.query(s_mod_norm, target, kind="record")
    
    s_trace = rec.trace
    d_trace = mutate_to_dyn_m(s_trace, 
        name_map={
            "matmul_reindex_shared.dyn_wmma.accumulator_o": "matmul_reindex_pad_shared.dyn_wmma.accumulator_o",
            "reshape_in_reindex_shared.dyn_wmma.matrix_a_o": "reshape_in_reindex_pad_shared.dyn_wmma.matrix_a_o"
        }
    )

    sch = tvm.tir.Schedule(d_mod)
    d_trace.apply_to_schedule(sch, remove_postproc=False)

    with target:
        dyn_m_lib = tvm.build(sch.mod["main"])


    B, M, N, K, G = 1, 1, 5120, 20480, 160
    args_info = [
        ms.arg_info.TensorInfo(shape=[B, M, K], dtype="float16"),
        ms.arg_info.TensorInfo(shape=[K//8, N], dtype="int32"),
        ms.arg_info.TensorInfo(shape=[G, N//8], dtype="int32"),
        ms.arg_info.TensorInfo(shape=[G, N], dtype="float16"),
        ms.arg_info.TensorInfo(shape=[N], dtype="float16"),
        ms.arg_info.TensorInfo(shape=[B, M, N], dtype="float16"),
    ]
    args = [generate_arg(info, dev) for info in args_info]

    score_s = dyn_m_lib.time_evaluator(dyn_m_lib.entry_name, dev=dev, number=2000, repeat=1)(*args).mean 
    print(f"[M:{M}] Dyn_M Duration {score_s * 1e6} us  (declared {float(rec.run_secs[0]) * 1e6} us)")
    

    new_db = ms.database.JSONDatabase(work_dir=dyn_db_name)
    d_workload = new_db.commit_workload(ms.tir_integration._normalize_mod(d_mod))
    d_rec = ms.database.TuningRecord(d_trace, d_workload, run_secs=[score_s])
    new_db.commit_tuning_record(d_rec)

if __name__ == "__main__":
    main()
