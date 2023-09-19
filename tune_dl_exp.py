import numpy as np

import tvm
import tvm.meta_schedule as ms
from tvm import te
from tvm._ffi import get_global_func

import tvm.script.tir as T
from tvm.tir.schedule import BlockRV, Schedule

import tvm.tir.tensor_intrin.cuda

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)


@T.prim_func
def matmul_from_mlc_llm_1(lv3: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), lv4: T.Buffer((T.int64(12288), T.int64(128)), "float16"), lv1615: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(12288)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3[v_i, v_j // T.int64(8)], lv4[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv3[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4[v_i, v_j // T.int64(32)]
    for i1, i2, k in T.grid(T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i1, v_i2, v_k = T.axis.remap("SSR", [i1, i2, k])
            T.reads(lv1615[0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[0, v_i1, v_i2] = var_NT_matmul_intermediate[0, v_i1, v_i2] + lv1615[0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]


@T.prim_func
def matmul_from_mlc_llm_2(lv16: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv17: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv16[v_i, v_j // T.int64(8)], lv17[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv16[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv17[v_i, v_j // T.int64(32)]
    for i1, i2, k in T.grid(T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i1, v_i2, v_k = T.axis.remap("SSR", [ i1, i2, k])
            T.reads(lv1654[0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[0, v_i1, v_i2] = var_NT_matmul_intermediate[0, v_i1, v_i2] + lv1654[0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def matmul_from_mlc_llm_2_m2(lv16: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv17: T.Buffer((T.int64(22016), T.int64(128)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(2), T.int64(4096)), "float16"), var_NT_matmul_intermediate: T.Buffer((T.int64(1), T.int64(2), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv16[v_i, v_j // T.int64(8)], lv17[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv16[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv17[v_i, v_j // T.int64(32)]
    for i1, i2, k in T.grid(T.int64(2), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i1, v_i2, v_k = T.axis.remap("SSR", [ i1, i2, k])
            T.reads(lv1654[0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[0, v_i1, v_i2] = var_NT_matmul_intermediate[0, v_i1, v_i2] + lv1654[0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]



def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)


def apply_trace(sch: tvm.tir.Schedule) -> None:
    b0 = sch.get_block(name="root", func_name="main")
    b1, b2 = sch.get_child_blocks(b0)
    sch.transform_block_layout(block=b1, index_map=lambda v_i, v_j: (v_i, v_j,))
    l3, l4 = sch.get_loops(block=b1)
    sch.transform_block_layout(block=b2, index_map=lambda v_i1, v_i2, v_k: (v_i2, v_k,))
    l5, l6 = sch.get_loops(block=b2)
    sch.compute_inline(block=b1)
    l7, l8 = sch.split(loop=l6, factors=[None, 8], preserve_unit_iters=True)
    l9 = sch.add_unit_loop(block_or_loop=b2)
    sch.reorder(l9, l5, l7, l8)
    l10 = sch.fuse(l9, preserve_unit_iters=True)
    l11 = sch.fuse(l5, preserve_unit_iters=True)
    l12 = sch.fuse(l7, preserve_unit_iters=True)
    l13, l14, l15, l16 = sch.get_loops(block=b2)
    l17, l18, l19, l20 = sch.get_loops(block=b2)
    l21 = sch.fuse(l17, l18, preserve_unit_iters=True)
    l22 = sch.fuse(l19, l20, preserve_unit_iters=True)
    l23, l24, l25 = sch.split(loop=l21, factors=[None, 4, 1], preserve_unit_iters=True)
    l26, l27, l28, l29 = sch.split(loop=l22, factors=[None, 64, 2, 4], preserve_unit_iters=True)
    sch.reorder(l26, l28, l27, l29)
    l30 = sch.fuse(l27, l29, preserve_unit_iters=True)
    b31 = sch.rfactor(loop=l30, factor_axis=0)
    l32, l33, l34, l35 = sch.get_loops(block=b2)
    l36, l37 = sch.split(loop=l35, factors=[64, None], preserve_unit_iters=True)
    b38 = sch.rfactor(loop=l36, factor_axis=0)
    l39, l40, l41, l42, l43, l44 = sch.get_loops(block=b31)
    l45, l46 = sch.split(loop=l44, factors=[64, None], preserve_unit_iters=True)
    sch.reorder(l39, l40, l45, l42, l41, l43, l46)
    sch.bind(loop=l39, thread_axis="blockIdx.x")
    sch.bind(loop=l40, thread_axis="threadIdx.y")
    sch.bind(loop=l45, thread_axis="threadIdx.x")
    sch.vectorize(loop=l46)
    b47 = sch.cache_read(block=b31, read_buffer_index=1, storage_scope="local")
    sch.compute_at(block=b47, loop=l42, preserve_unit_loops=True, index=-1)
    l48, l49, l50, l51, l52, l53 = sch.get_loops(block=b47)
    l54, l55 = sch.split(loop=l52, factors=[None, 1], preserve_unit_iters=True)
    sch.reorder(l54, l53, l55)
    sch.vectorize(loop=l55)
    b56 = sch.cache_read(block=b31, read_buffer_index=0, storage_scope="shared")
    sch.compute_at(block=b56, loop=l45, preserve_unit_loops=True, index=-1)
    l57, l58, l59, l60, l61, l62 = sch.get_loops(block=b56)
    l63, l64, l65, l66 = sch.split(loop=l62, factors=[None, 4, 64, 8], preserve_unit_iters=True)
    sch.bind(loop=l64, thread_axis="threadIdx.y")
    sch.bind(loop=l65, thread_axis="threadIdx.x")
    sch.vectorize(loop=l66)
    sch.reverse_compute_at(block=b38, loop=l39, preserve_unit_loops=True, index=-1)
    l67, l68, l69, l70 = sch.get_loops(block=b38)
    l71 = sch.fuse(l70, preserve_unit_iters=True)
    l72, l73 = sch.split(loop=l71, factors=[4, None], preserve_unit_iters=True)
    l74, l75 = sch.split(loop=l73, factors=[None, 1], preserve_unit_iters=True)
    sch.reorder(l72, l68, l74, l75, l69)
    sch.bind(loop=l72, thread_axis="threadIdx.y")
    sch.bind(loop=l68, thread_axis="threadIdx.x")
    sch.vectorize(loop=l75)
    sch.reverse_compute_at(block=b2, loop=l39, preserve_unit_loops=True, index=-1)
    l76, l77, l78 = sch.get_loops(block=b2)
    l79 = sch.fuse(l78, preserve_unit_iters=True)
    l80, l81 = sch.split(loop=l79, factors=[4, None], preserve_unit_iters=True)
    sch.reorder(l81, l80, l77)
    sch.bind(loop=l80, thread_axis="threadIdx.y")
    sch.bind(loop=l77, thread_axis="threadIdx.x")
    l82, l83, l84, l85, l86, l87, l88 = sch.get_loops(block=b31)
    b89 = sch.decompose_reduction(block=b31, loop=l85)
    l90, l91, l92, l93, l94, l95 = sch.get_loops(block=b38)
    b96 = sch.decompose_reduction(block=b38, loop=l95)
    sch.set_scope(block=b31, buffer_index=0, storage_scope="local")
    sch.set_scope(block=b38, buffer_index=0, storage_scope="local")
    l97, l98, l99, l100, l101, l102, l103 = sch.get_loops(block=b31)
    sch.annotate(block_or_loop=l100, ann_key="pragma_auto_unroll_max_step", ann_val=256)
    l104, l105, l106, l107, l108, l109, l110 = sch.get_loops(block=b31)
    sch.annotate(block_or_loop=l107, ann_key="pragma_unroll_explicit", ann_val=1)
    l111, l112, l113, l114, l115, l116 = sch.get_loops(block=b38)
    sch.annotate(block_or_loop=l114, ann_key="pragma_auto_unroll_max_step", ann_val=256)
    l117, l118, l119, l120, l121, l122 = sch.get_loops(block=b38)
    sch.annotate(block_or_loop=l120, ann_key="pragma_unroll_explicit", ann_val=1)
    l123, l124, l125, l126, l127, l128, l129, l130, l131 = sch.get_loops(block=b56)
    sch.annotate(block_or_loop=l128, ann_key="pragma_unroll_explicit", ann_val=256)
    l132, l133, l134, l135, l136, l137, l138, l139, l140 = sch.get_loops(block=b56)
    sch.annotate(block_or_loop=l137, ann_key="pragma_vectorize", ann_val=1)


def check_dl():
    # func = matmul_from_mlc_llm_1
    func = matmul_from_mlc_llm_2
    # func = matmul_from_mlc_llm_2_m2
    mod = tvm.IRModule({"main": func})

    with TARGET:
        # import tvm.dlight as dl
        # dl_mod = dl.ApplyDefaultSchedule(dl.gpu.GEMV())(mod)
        # dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Matmul())(dl_mod)
        # dl_mod = dl.ApplyDefaultSchedule(dl.gpu.Fallback())(dl_mod)
        # lib = tvm.build(dl_mod)

        sch = tvm.tir.Schedule(mod)
        apply_trace(sch)
        lib = tvm.build(sch.mod)

    args_info = ms.arg_info.ArgInfo.from_prim_func(func)
    args = [make_arg(info) for info in args_info]

    score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=2000, repeat=1)(*args).mean
    print(f"SCORE : {score_s*1e6} us",)


if __name__ == "__main__":
    check_dl()
