import numpy as np
import subprocess
from pathlib import Path

import tvm
import tvm.script.tir as T
from tvm.contrib import nvcc

import tvm.tir.tensor_intrin.cuda

TARGET = tvm.target.Target("nvidia/nvidia-a10g")
DEV = tvm.cuda(0)

def make_arg(info):
    if info.dtype in ["float16", "float32"]:
        arr_np = np.random.uniform(-1, 1, size=info.shape).astype(info.dtype)
    elif info.dtype in ["int32", "uint32", "int16", "int8"]:
        arr_np = np.random.randint(0, 16, size=info.shape).astype(info.dtype)
    else:
        assert False, f"Unimplemented, dtype={info.dtype}"

    return tvm.nd.array(arr_np, device=DEV)

def get_sass(cubin):
    temp = utils.tempdir()
    temp_cubin = temp.relpath("my_kernel.cubin")
    with open(temp_cubin, "wb") as out_file:
        out_file.write(cubin)
    
    cmd = [ "nvdisasm", "-c", temp_cubin]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if proc.returncode != 0:
        msg += "\nCompilation error:\n"
        msg += out.decode("utf-8")
        raise RuntimeError(msg)

    return out.decode("utf-8")


def cuda_dump(lib, dump_path="."):
    src = lib.imported_modules[0].get_source()
    with open(f"{dump_path}/shaders.cu", "w") as f:
        print(src, file=f)

    ptx = nvcc.compile_cuda(src, target_format="ptx")
    with open(f"{dump_path}/shaders.ptx", "wb") as f:
        f.write(ptx)

    cubin = nvcc.compile_cuda(src, target_format="cubin")
    # with open(f"{dump_path}/shaders.cubin", "wb") as f:
        # f.write(cubin)

    sass = get_sass(cubin)
    with open(f"{dump_path}/shaders.sass", "w") as f:
        f.write(sass)


# q4f16_3
@T.prim_func
def matmul_g128_KN_sym_dynm(lv503: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv504: T.Buffer((T.int64(32), T.int64(22016)), "float16"), lv1654_hdl: T.handle, var_matmul_intermediate_hdl: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    m = T.int64()
    lv1654 = T.match_buffer(lv1654_hdl, (T.int64(1), m, T.int64(4096)), "float16")
    var_matmul_intermediate = T.match_buffer(var_matmul_intermediate_hdl, (T.int64(1), m, T.int64(22016)), "float16")

    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i // T.int64(8), v_j], lv504[v_i // T.int64(128), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            # Original zp FP16
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv503[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv504[v_i // T.int64(128), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), m, T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]


#
# From mlc-llm Vicuna 7B q4f16
# 
@T.prim_func
def fused_fused_decode11_fused_matmul10_add1(lv639: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv640: T.Buffer((T.int64(344), T.int64(4096)), "float16"), lv638: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv634: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv639[v_i // T.int64(8), v_j], lv640[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv639[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv640[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv638[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv638[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv634[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv634[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode10_matmul9(lv635: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv636: T.Buffer((T.int64(128), T.int64(22016)), "float16"), lv1654: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv635[v_i // T.int64(8), v_j], lv636[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv635[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv636[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1654[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1654[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]


@T.prim_func
def matmul7(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]


@T.prim_func
def fused_fused_decode8_matmul6(lv619: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), lv620: T.Buffer((T.int64(128), T.int64(12288)), "float16"), lv1615: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_matmul_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(12288)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv619[v_i // T.int64(8), v_j], lv620[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv619[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv620[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1615[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1615[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]

@T.prim_func
def fused_NT_matmul5_divide2_maximum1_minimum1_cast3(lv1637: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), p_lv1638: T.handle, p_lv1614: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv1638 = T.match_buffer(p_lv1638, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    lv1614 = T.match_buffer(p_lv1614, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv1637[v_i0, v_i1, v_i2, v_k], lv1638[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv1637[v_i0, v_i1, v_i2, v_k] * lv1638[v_i0, v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_minimum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1614[v_ax0, T.int64(0), v_ax2, v_ax3])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])


@T.prim_func
def fused_NT_matmul1_divide_maximum_minimum_cast(p_lv28: T.handle, p_lv29: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv28 = T.match_buffer(p_lv28, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    m = T.int64()
    lv29 = T.match_buffer(p_lv29, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
    var_compute_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m))
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_maximum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    var_T_minimum_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, m, T.int64(128)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv28[v_i0, v_i1, v_i2, v_k], lv29[v_i0, v_i1, v_i3, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv28[v_i0, v_i1, v_i2, v_k] * lv29[v_i0, v_i1, v_i3, v_k]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_maximum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.max(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], T.float16(-65504))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_minimum"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_minimum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.min(var_T_maximum_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.Cast("float32", var_T_minimum_intermediate[v_i0, v_i1, v_i2, v_i3])


@T.prim_func
def fused_fused_decode3_fused_NT_matmul2_add(lv11: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv12: T.Buffer((T.int64(128), T.int64(4096)), "float16"), p_lv41: T.handle, p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv11[v_i // T.int64(8), v_j], lv12[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv11[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv12[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv41[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv41[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode9_fused_matmul8_add1(lv631: T.Buffer((T.int64(512), T.int64(4096)), "uint32"), lv632: T.Buffer((T.int64(128), T.int64(4096)), "float16"), lv630: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), lv1613: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv631[v_i // T.int64(8), v_j], lv632[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv631[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv632[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv630[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv630[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv1613[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv1613[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def matmul(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    matmul = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), m):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def fused_fused_decode4_NT_matmul3(lv15: T.Buffer((T.int64(512), T.int64(22016)), "uint32"), lv16: T.Buffer((T.int64(128), T.int64(22016)), "float16"), p_lv45: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv45 = T.match_buffer(p_lv45, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(22016)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv15[v_i // T.int64(8), v_j], lv16[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv15[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv16[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate[v_ax0, v_ax1])
            p_output0_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv45[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv45[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_fused_decode5_fused_NT_matmul4_add(lv19: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"), lv20: T.Buffer((T.int64(344), T.int64(4096)), "float16"), p_lv18: T.handle, p_lv14: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv18 = T.match_buffer(p_lv18, (T.int64(1), n, T.int64(11008)), "float16")
    lv14 = T.match_buffer(p_lv14, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv19[v_i // T.int64(8), v_j], lv20[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv19[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv20[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate_1[v_ax0, v_ax1])
            p_output0_intermediate_1[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv18[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv18[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv14[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv14[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]



@T.prim_func
def fused_fused_decode7_fused_matmul1_cast2(p_lv612: T.handle, p_lv613: T.handle, lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    vocab_size = T.int64()
    lv612 = T.match_buffer(p_lv612, (T.int64(512), vocab_size), "uint32")
    lv613 = T.match_buffer(p_lv613, (T.int64(128), vocab_size), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), vocab_size))
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), vocab_size), "float16")
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), vocab_size), "float16")
    for i, j in T.grid(T.int64(4096), vocab_size):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv612[v_i // T.int64(8), v_j], lv613[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv612[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv613[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), vocab_size, T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), vocab_size):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_matmul_intermediate[v_i0, v_i1, v_i2])


@T.prim_func
def fused_fused_decode2_NT_matmul(lv4: T.Buffer((T.int64(512), T.int64(12288)), "uint32"), lv5: T.Buffer((T.int64(128), T.int64(12288)), "float16"), p_lv6: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(12288)), "float16")
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv4[v_i // T.int64(8), v_j], lv5[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv4[v_i // T.int64(8), v_j], T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv5[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(p_output0_intermediate[v_ax0, v_ax1])
            p_output0_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv6[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv6[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]


decoder_functions = [
    fused_fused_decode8_matmul6,
    fused_fused_decode9_fused_matmul8_add1,
    fused_fused_decode10_matmul9,
    fused_fused_decode11_fused_matmul10_add1,
    #fused_NT_matmul5_divide2_maximum1_minimum1_cast3  # to Masa
]

prefill_functions = [
    fused_fused_decode2_NT_matmul,
    fused_fused_decode3_fused_NT_matmul2_add,
    fused_fused_decode4_NT_matmul3,
    fused_fused_decode5_fused_NT_matmul4_add,
    #fused_NT_matmul1_divide_maximum_minimum_cast  # to Masa
]

decoder_mod = tvm.IRModule({func.__name__:func for func in decoder_functions})
prefill_mod = tvm.IRModule({func.__name__:func for func in prefill_functions})
