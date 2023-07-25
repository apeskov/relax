
import tvm
from tvm.script import tir as T
import tvm.meta_schedule as ms
from tvm import te

import tvm.tir.tensor_intrin.cuda

def construct_te(B, M, N, K, G):
    # n = te.var("n")
    # B, M, N, K, G = 1, 1, 5120, 5120, 40
    GS = K // G
    
    if B is None:
        a = te.placeholder((M, K), name="A", dtype="float16")
    else:
        a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K//8, N), name="qwgh", dtype="int32")
    scl = te.placeholder((G, N), name="scl", dtype="float16")
    qzp = te.placeholder((G, N//8), name="qzp", dtype="int32")
    bias = te.placeholder((N,), name="bias", dtype="float16")

    decoded_zp = te.compute(
        (qzp.shape[0], qzp.shape[1]*8),
        lambda vg, vn: 
            (qzp[vg, vn // 8] >> (vn % 8 * 4) & 0xF).astype("int32") + tvm.tir.const(1, dtype="int32"),
        name="decode_zp"
        )

    decoded_wgh = te.compute(
        (qwgh.shape[0] * 8, qwgh.shape[1]),
        lambda vk, vn: 
            ((qwgh[vk // 8, vn] >> (vk % 8 * 4) & 0xF) - decoded_zp[vk // GS, vn]).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh"
        )

    k = te.reduce_axis((0, decoded_wgh.shape[0]), name="k")
    if B is None:
        x = te.compute(
            (a.shape[0], qwgh.shape[1]),
            lambda vm, vn: te.sum(a[vm, k] * decoded_wgh[k, vn], axis=k),
            name="matmul"
        )
    else:
        x = te.compute(
            (a.shape[0], a.shape[1], qwgh.shape[1]),
            lambda vb, vm, vn: te.sum(a[vb, vm, k] * decoded_wgh[k, vn], axis=k),
            name="matmul"
        )

    # x = x + bias
    
    return tvm.te.create_prim_func([a, qwgh, scl, qzp, bias, x])


def construct_te_2(B, M, N, K, G):
    GS = K // G
    
    # Convert to int64
    B = tvm.tir.const(B, dtype="int64")
    M = tvm.tir.const(M, dtype="int64")
    N = tvm.tir.const(N, dtype="int64")
    K = tvm.tir.const(K, dtype="int64")
    G = tvm.tir.const(G, dtype="int64")
    GS = tvm.tir.const(GS, dtype="int64")

    a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K//8, N), name="qwgh", dtype="int32")
    scl = te.placeholder((G, N), name="scl", dtype="float16")
    qzp = te.placeholder((G, N//8), name="qzp", dtype="int32")
    bias = te.placeholder((N,), name="bias", dtype="float16")
    
    decoded_zp = te.compute(
        (qzp.shape[0], qzp.shape[1]*8),
        lambda vg, vn: 
            (qzp[vg, vn // 8] >> (vn.astype("int32") % 8 * 4) & 0xF) + 1,
            name="decode_zp"
        )

    decoded_wgh = te.compute(
        (qwgh.shape[0] * 8, qwgh.shape[1]),
        lambda vk, vn: 
            ((qwgh[vk // 8, vn] >> (vk.astype("int32") % 8 * 4) & 0xF) - decoded_zp[vk // GS, vn]).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh"
        )
    
    # Reshape in. WA
    x = te.compute(
        (a.shape[1], a.shape[2]),
        lambda vm, vk: a[0, vm, vk],
        name="reshape_in"
    )

    k = te.reduce_axis((0, decoded_wgh.shape[0]), name="k")                
    x = te.compute(
        (x.shape[0], qwgh.shape[1]),
        lambda vm, vn: te.sum(x[vm, k] * decoded_wgh[k, vn], axis=k),
        name="matmul"
    )

    # Reshape out, WA
    x = te.compute(
        (1, x.shape[0], x.shape[1]),
        lambda vb, vm, vn: x[vm, vn],
        name="reshape_out"
    )

    # if bias is not None:
        # x = x + bias
    
    return tvm.te.create_prim_func([a, qwgh, scl, qzp, bias, x])

#
#  Good version of q_mamtul_5120_20480 with GELU ???
#  Looks like a better version of previous one. 
#  On TVM demonstrate 121 us 
#
@T.prim_func
def fused_q_matmul4_gelu1_32(lv1822: T.Buffer((T.int64(1), T.int64(32), T.int64(5120)), "float16"), linear_weight221: T.Buffer((T.int64(640), T.int64(20480)), "int32"), linear_zp220: T.Buffer((T.int64(40), T.int64(2560)), "int32"), linear_scl220: T.Buffer((T.int64(40), T.int64(20480)), "float16"), linear_bias220: T.Buffer((T.int64(20480),), "float16"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    reshape_in = T.alloc_buffer((T.int64(32), T.int64(5120)), "float16")
    decode_zp = T.alloc_buffer((T.int64(40), T.int64(20480)), "int32")
    decode_wgh = T.alloc_buffer((T.int64(5120), T.int64(20480)), "float16")
    matmul = T.alloc_buffer((T.int64(1), T.int64(20480)), "float16")
    reshape_out = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    T_multiply = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    compute = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)))
    compute_1 = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)))
    compute_2 = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    T_add = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16")
    for vm, vk in T.grid(T.int64(32), T.int64(5120)):
        with T.block("reshape_in"):
            v_vm, v_vk = T.axis.remap("SS", [vm, vk])
            T.reads(lv1822[T.int64(0), v_vm, v_vk])
            T.writes(reshape_in[v_vm, v_vk])
            reshape_in[v_vm, v_vk] = lv1822[T.int64(0), v_vm, v_vk]
    for vg, vn in T.grid(T.int64(40), T.int64(20480)):
        with T.block("decode_zp"):
            v_vg, v_vn = T.axis.remap("SS", [vg, vn])
            T.reads(linear_zp220[v_vg, v_vn // T.int64(8)])
            T.writes(decode_zp[v_vg, v_vn])
            decode_zp[v_vg, v_vn] = T.bitwise_and(T.shift_right(linear_zp220[v_vg, v_vn // T.int64(8)], T.Cast("int32", v_vn) % 8 * 4), 15) + 1
    for vk, vn in T.grid(T.int64(5120), T.int64(20480)):
        with T.block("decode_wgh"):
            v_vk, v_vn = T.axis.remap("SS", [vk, vn])
            T.reads(linear_weight221[v_vk // T.int64(8), v_vn], decode_zp[v_vk // T.int64(128), v_vn], linear_scl220[v_vk // T.int64(128), v_vn])
            T.writes(decode_wgh[v_vk, v_vn])
            decode_wgh[v_vk, v_vn] = T.Cast("float16", T.bitwise_and(T.shift_right(linear_weight221[v_vk // T.int64(8), v_vn], T.Cast("int32", v_vk) % 8 * 4), 15) - decode_zp[v_vk // T.int64(128), v_vn]) * linear_scl220[v_vk // T.int64(128), v_vn]
    for vm, vn, k in T.grid(T.int64(32), T.int64(20480), T.int64(5120)):
        with T.block("matmul"):
            v_vm, v_vn, v_k = T.axis.remap("SSR", [vm, vn, k])
            T.reads(reshape_in[v_vm, v_k], decode_wgh[v_k, v_vn])
            T.writes(matmul[v_vm, v_vn])
            with T.init():
                matmul[v_vm, v_vn] = T.float16(0)
            matmul[v_vm, v_vn] = matmul[v_vm, v_vn] + reshape_in[v_vm, v_k] * decode_wgh[v_k, v_vn]
    for vb, vm, vn in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("reshape_out"):
            v_vb, v_vm, v_vn = T.axis.remap("SSS", [vb, vm, vn])
            T.reads(matmul[v_vm, v_vn])
            T.writes(reshape_out[v_vb, v_vm, v_vn])
            reshape_out[v_vb, v_vm, v_vn] = matmul[v_vm, v_vn]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(reshape_out[v_ax0, v_ax1, v_ax2], linear_bias220[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = reshape_out[v_ax0, v_ax1, v_ax2] + linear_bias220[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float16(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.Cast("float32", T_multiply[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("compute_1"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(compute[v_i0, v_i1, v_i2])
            T.writes(compute_1[v_i0, v_i1, v_i2])
            compute_1[v_i0, v_i1, v_i2] = T.erf(compute[v_i0, v_i1, v_i2])
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("compute_2"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(compute_1[v_i0, v_i1, v_i2])
            T.writes(compute_2[v_i0, v_i1, v_i2])
            compute_2[v_i0, v_i1, v_i2] = T.Cast("float16", compute_1[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute_2[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute_2[v_ax0, v_ax1, v_ax2] * T.float16(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float16(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(20480)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]


@T.prim_func
def q_matmul5(A: T.Buffer((T.int64(1), T.int64(1), T.int64(20480)), "float16"), B: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), C: T.Buffer((T.int64(160), T.int64(640)), "int32"), D: T.Buffer((T.int64(160), T.int64(5120)), "float16"), E: T.Buffer((T.int64(5120),), "float16"), T_add: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    reshape_in = T.alloc_buffer((T.int64(1), T.int64(20480)), "float16")
    decode_zp = T.alloc_buffer((T.int64(160), T.int64(5120)), "int32")
    decode_wgh = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16")
    matmul = T.alloc_buffer((T.int64(1), T.int64(5120)), "float16")
    reshape_out = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16")
    for vm, vk in T.grid(T.int64(1), T.int64(20480)):
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
    for vm, vn, k in T.grid(T.int64(1), T.int64(5120), T.int64(20480)):
        with T.block("matmul"):
            v_vm, v_vn, v_k = T.axis.remap("SSR", [vm, vn, k])
            T.reads(reshape_in[v_vm, v_k], decode_wgh[v_k, v_vn])
            T.writes(matmul[v_vm, v_vn])
            with T.init():
                matmul[v_vm, v_vn] = T.float16(0)
            matmul[v_vm, v_vn] = matmul[v_vm, v_vn] + reshape_in[v_vm, v_k] * decode_wgh[v_k, v_vn]
    for vb, vm, vn in T.grid(T.int64(1), T.int64(1), T.int64(5120)):
        with T.block("reshape_out"):
            v_vb, v_vm, v_vn = T.axis.remap("SSS", [vb, vm, vn])
            T.reads(matmul[v_vm, v_vn])
            T.writes(reshape_out[v_vb, v_vm, v_vn])
            reshape_out[v_vb, v_vm, v_vn] = matmul[v_vm, v_vn]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(reshape_out[v_ax0, v_ax1, v_ax2], E[v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = reshape_out[v_ax0, v_ax1, v_ax2] + E[v_ax2]


@T.prim_func
def q_matmul5_32(A: T.Buffer((T.int64(1), T.int64(32), T.int64(20480)), "float16"), B: T.Buffer((T.int64(2560), T.int64(5120)), "int32"), C: T.Buffer((T.int64(160), T.int64(640)), "int32"), D: T.Buffer((T.int64(160), T.int64(5120)), "float16"), E: T.Buffer((T.int64(5120),), "float16"), T_add: T.Buffer((T.int64(1), T.int64(32), T.int64(5120)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    reshape_in = T.alloc_buffer((T.int64(32), T.int64(20480)), "float16")
    decode_zp = T.alloc_buffer((T.int64(160), T.int64(5120)), "int32")
    decode_wgh = T.alloc_buffer((T.int64(20480), T.int64(5120)), "float16")
    matmul = T.alloc_buffer((T.int64(32), T.int64(5120)), "float16")
    reshape_out = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(5120)), "float16")
    for vm, vk in T.grid(T.int64(32), T.int64(20480)):
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
    for vm, vn, k in T.grid(T.int64(32), T.int64(5120), T.int64(20480)):
        with T.block("matmul"):
            v_vm, v_vn, v_k = T.axis.remap("SSR", [vm, vn, k])
            T.reads(reshape_in[v_vm, v_k], decode_wgh[v_k, v_vn])
            T.writes(matmul[v_vm, v_vn])
            with T.init():
                matmul[v_vm, v_vn] = T.float16(0)
            matmul[v_vm, v_vn] = matmul[v_vm, v_vn] + reshape_in[v_vm, v_k] * decode_wgh[v_k, v_vn]
    for vb, vm, vn in T.grid(T.int64(1), T.int64(32), T.int64(5120)):
        with T.block("reshape_out"):
            v_vb, v_vm, v_vn = T.axis.remap("SSS", [vb, vm, vn])
            T.reads(matmul[v_vm, v_vn])
            T.writes(reshape_out[v_vb, v_vm, v_vn])
            reshape_out[v_vb, v_vm, v_vn] = matmul[v_vm, v_vn]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(reshape_out[v_ax0, v_ax1, v_ax2], E[v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = reshape_out[v_ax0, v_ax1, v_ax2] + E[v_ax2]


def construct_r_fuctor_te_v2(M, N, K, G, RF):
    GS = K // G

    a = te.placeholder((1, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K//8, N), name="qwgh", dtype="int32")
    scl = te.placeholder((G, N), name="scl", dtype="float16")
    qzp = te.placeholder((G, N//8), name="qzp", dtype="int32")

    # Decode weights
    decoded_zp = te.compute(
        (qzp.shape[0], qzp.shape[1]*8),
        lambda vg, vn: 
            (qzp[vg, vn // 8] >> (vn.astype("int32") % 8 * 4) & 0xF) + 1,
        name="decode_zp",
    )
    wgh = te.compute(
        (qwgh.shape[0] * 8, qwgh.shape[1]),
        lambda vk, vn: 
            ((qwgh[vk // 8, vn] >> (vk.astype("int32") % 8 * 4) & 0xF) - decoded_zp[vk // GS, vn]).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh",
    )

    # Reshape A in (M, RF, K//RF). WA2                 
    x = te.compute(
        (M, RF, K//RF),
        lambda vm, kb, vk: a[0, vm, vk + kb * K//RF],
        name="reshape_in"
    )

    # Reshape B to (RF, K//RF, N)
    wgh= te.compute(
        (RF , K//RF, N),
        lambda kb, vk, vn: wgh[vk + kb * K//RF, vn],
        name="resh_b"
    )

    # Batched matmul
    k = te.reduce_axis((0, K//RF), name="k")
    x = te.compute(
        (RF, M, N),
        lambda kb, vm, vn: te.sum(x[vm, kb, k] * wgh[kb, k, vn], axis=k),
        name="matmul"
    )

    rf = te.reduce_axis((0, RF), name="k")
    x = te.compute(
        (1, M, N),
        lambda _, vm, vn: te.sum(x[rf, vm, vn], axis=rf),
        name=f"reduction_{RF}"
    )
    return tvm.te.create_prim_func([a, qwgh, scl, qzp, x])

def construct_simple_te(M, N, K, B):
    a = te.placeholder((B, M, K), name="A", dtype="float16")
    b = te.placeholder((B, K, N), name="B", dtype="float16")

    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (B, M, N),
        lambda vb, vm, vn: te.sum(a[vb, vm, k] * b[vb, k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, b, x])


def main():
    target = tvm.target.Target("nvidia/nvidia-a10g")
    work_dir = "dolly_tune_tir_4b"

    # ms_rule_type = "cuda"
    ms_rule_type = "cuda-tensorcore"
    
    HS = 5120
    HS4 = HS * 4
    #                                 BN    M   N     K     G
    # funcs = {"q_matmul": construct_te(None, 1, HS,   HS,   HS//128)}
    # funcs = {"q_matmul": construct_te(None, 32, HS*3, HS,   HS//128)}
    # funcs = {"q_matmul": construct_te(None, 1, HS*4, HS,   HS//128)}
    # funcs = {"q_matmul": construct_te(None, 1, HS  , HS*4, HS*4//128)}
    
    # funcs = {"fused_q_matmul4_gelu1": fused_q_matmul4_gelu1}
    
    # funcs = {"fused_q_matmul4_gelu1_32": fused_q_matmul4_gelu1_32}
    funcs = {
        # "q_matmul5_32": q_matmul5_32,
        # "q_matmul_r4_1_HS_4HS":    construct_r_fuctor_te_v2(32, HS, HS4, 160, RF=4),
        "simple_matmul_128":    construct_simple_te(128, 128, 128, B=2),
    }
    
    # funcs = {"construct_te_2_1_32_15360_5120": construct_te_2(1, 32, HS*3, HS,   HS//128)}
    # funcs = {"construct_te_2_1_32_5120_20480": construct_te_2(1, 32, HS, HS*4, HS*4//128)}

    database = ms.tir_integration.tune_tir(
        mod=tvm.ir.IRModule(funcs),
        target=target,
        work_dir=work_dir,
        max_trials_global=100500,
        max_trials_per_task=2048,
        num_trials_per_iter=32,
        # max_trials_per_task=32,
        # num_trials_per_iter=8,
        space=ms.space_generator.PostOrderApply(
                sch_rules=ms_rule_type,
                postprocs=ms_rule_type,
                mutator_probs=ms_rule_type,
            ),
    )

if __name__ == "__main__":
    main()
