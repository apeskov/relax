import tvm
from tvm import te

def simple_mm(M, N, K):
    a = te.placeholder([M, K], dtype="float16", name="A")
    b = te.placeholder([K, N], dtype="float16", name="B")
    
    k = te.reduce_axis([0, K], name="k")
    x = te.compute(
        [M, N],
        lambda m, n: te.sum(a[m, k] * b[k, n], axis=k),
        name="matmul"
    )
    return te.create_prim_func([a, b, x])


def apply_trace_man(sch: tvm.tir.Schedule) -> None: 
    block = sch.get_block("matmul")
    lm, ln, lk = sch.get_loops(block)

    lm_5, lm_4, lm_3, lm_2, lm_1, lm_0 = sch.split(lm, factors=[None, 2, 1, 1, 1, 2])
    ln_5, ln_4, ln_3, ln_2, ln_1, ln_0 = sch.split(ln, factors=[1, None, 4, 1, 4, 2])
    lk_1, lk_0 = sch.split(lk, factors=[None, 4])
    sch.reorder(lm_5, ln_5, lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lm_1, ln_1, lk_1, lk_0, lm_0, ln_0)
    
    sch.bind(ln_3, thread_axis="threadIdx.y")
    
    b_b_shared = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_shared, loop=lk_1, preserve_unit_loops=True, index=-1)
    
    # sch.bind(ln_3, thread_axis="threadIdx.y")  # << Error is here

    print("!! All right !!")


def test_wrong_shape():
    M, N, K = 1024, 1024, 1024
    func = simple_mm(M, N, K)
    sch = tvm.tir.Schedule(func, debug_mask="all")

    block = sch.get_block("matmul")
    lm, ln, lk = sch.get_loops(block)

    lm_5, lm_4, lm_3, lm_2, lm_1, lm_0 = sch.split(lm, factors=[None, 2, 1, 1, 1, 2])
    ln_5, ln_4, ln_3, ln_2, ln_1, ln_0 = sch.split(ln, factors=[1, None, 4, 1, 4, 2])
    lk_1, lk_0 = sch.split(lk, factors=[None, 4])
    sch.reorder(lm_5, ln_5, lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lm_1, ln_1, lk_1, lk_0, lm_0, ln_0)
    
    sch.bind(ln_3, thread_axis="threadIdx.y")
    
    b_b_shared = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_shared, loop=lk_1, preserve_unit_loops=True, index=-1)

    print(sch.mod)


def test_wrong_sch_state():
    M, N, K = 1024, 1024, 1024
    func = simple_mm(M, N, K)
    sch = tvm.tir.Schedule(func, debug_mask="all")

    block = sch.get_block("matmul")
    lm, ln, lk = sch.get_loops(block)

    lm_5, lm_4, lm_3, lm_2, lm_1, lm_0 = sch.split(lm, factors=[None, 2, 1, 1, 1, 2])
    ln_5, ln_4, ln_3, ln_2, ln_1, ln_0 = sch.split(ln, factors=[1, None, 4, 1, 4, 2])
    lk_1, lk_0 = sch.split(lk, factors=[None, 4])
    sch.reorder(lm_5, ln_5, lm_4, ln_4, lm_3, ln_3, lm_2, ln_2, lm_1, ln_1, lk_1, lk_0, lm_0, ln_0)
        
    b_b_shared = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_shared, loop=lk_1, preserve_unit_loops=True, index=-1)

    sch.bind(ln_3, thread_axis="threadIdx.y")


def test_wrong_sch_state_wa1():
    M, N, K = 1024, 1024, 1024
    func = simple_mm(M, N, K)
    sch = tvm.tir.Schedule(func, debug_mask="all")

    block = sch.get_block("matmul")
    lm, ln, lk = sch.get_loops(block)

    # Avoid usage intermediate loops between threads and place of cache read
    lm_5, lm_4, lm_3, lm_0 = sch.split(lm, factors=[None, 4, 4, 2])
    ln_5, ln_4, ln_3, ln_0 = sch.split(ln, factors=[None, 4, 4, 2])
    lk_1, lk_0 = sch.split(lk, factors=[None, 4])
    sch.reorder(lm_5, ln_5, lm_4, ln_4, lm_3, ln_3, lk_1, lk_0, lm_0, ln_0)
        
    b_b_cache = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_cache, loop=lk_1, preserve_unit_loops=True, index=-1)

    # But still has problem with order of bind calls
    # This order is correct   
    # sch.bind(ln_3, thread_axis="threadIdx.x")
    # sch.bind(ln_4, thread_axis="threadIdx.y")

    # This order leads to error   
    sch.bind(ln_4, thread_axis="threadIdx.y")
    sch.bind(ln_3, thread_axis="threadIdx.x")
    

def test_wrong_shape_2():
    M, N, K = 1024, 1024, 1024
    func = simple_mm(M, N, K)
    sch = tvm.tir.Schedule(func, debug_mask="all")

    block = sch.get_block("matmul")
    lm, ln, lk = sch.get_loops(block)

    # Avoid usage intermediate loops between threads and place of cache read
    lm_5, lm_4, lm_3, lm_0 = sch.split(lm, factors=[None, 4, 4, 2])
    ln_5, ln_4, ln_3, ln_0 = sch.split(ln, factors=[None, 4, 4, 2])
    lk_1, lk_0 = sch.split(lk, factors=[None, 4])
    sch.reorder(lm_5, ln_5, lm_4, ln_4, lm_3, ln_3, lk_1, lk_0, lm_0, ln_0)

    # But still has problem with order of bind calls
    # This order is correct   
    sch.bind(ln_3, thread_axis="threadIdx.x")
    sch.bind(ln_4, thread_axis="threadIdx.y")

    b_b_cache = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_cache, loop=lk_1, preserve_unit_loops=True, index=-1)

    print(sch.mod)


def test_wrong_state_simple():
    M = 1024
    a = te.placeholder([M], dtype="float16", name="A")
    b = te.placeholder([M], dtype="float16", name="B")
    x = te.compute([M], lambda m: a[m] * b[m], name="mul")
    
    func = te.create_prim_func([a, b, x])
    sch = tvm.tir.Schedule(func, debug_mask="all")

    # === Scheduling ===
    block = sch.get_block("mul")
    lm, = sch.get_loops(block)

    lm_bx, lm_ty, lm_tx, lm_X, lm_Y, lm_Z = sch.split(lm, factors=[None, 2, 2, 2, 2, 2])

    # This order is correct   
    sch.bind(lm_tx, thread_axis="threadIdx.x")
    sch.bind(lm_ty, thread_axis="threadIdx.y")

    b_b_cache = sch.cache_read(block, read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_cache, loop=lm_tx, preserve_unit_loops=True, index=-1)

    # Correct
    # sch.bind(lm_tx, thread_axis="threadIdx.x")
    # sch.bind(lm_ty, thread_axis="threadIdx.y")
    
    ## Incorrect
    # sch.bind(lm_ty, thread_axis="threadIdx.y")
    # sch.bind(lm_tx, thread_axis="threadIdx.x")

    print(sch.mod)


def test_simple():
    M = 1024
    a = te.placeholder([M], dtype="float16", name="A")
    b = te.placeholder([M], dtype="float16", name="B")
    x = te.compute([M], lambda m: a[m] + b[m], name="add")

    func = te.create_prim_func([a, b, x])
    sch = tvm.tir.Schedule(func, debug_mask="all")
    

    lm, = sch.get_loops("add")
    _, lm_1, lm_2, _ = sch.split(lm, factors=[None, 2, 2, 2])
    sch.reorder(lm_2, lm_1)

    b_b_cache = sch.cache_read("add", read_buffer_index=1, storage_scope="shared.dyn")
    sch.compute_at(b_b_cache, loop=lm_2, preserve_unit_loops=True, index=-1)
    
    print("""
          TVM is not able to correctly handle non contiguous regions.

          It leads to wrong shape of cache read. It should read 4 elements, but actually reads 6.
          Expected:  
          ...
          for ax0, ax1 in T.grid(2, 2):
                with T.block("B_shared.dyn"):
                    v0 = T.axis.spatial(1024, m_0 * 8 + m_2 * 2 + ax0 * 4 + ax1)
                    T.reads(B[v0])
                    T.writes(B_shared_dyn[v0])
                    B_shared_dyn[v0] = B[v0]
          
          Actual result:
          """)
    print(sch.mod)


if __name__ == "__main__":
    # test_wrong_shape()
    # test_wrong_sch_state()
    # test_wrong_sch_state_wa1()
    
    # test_wrong_shape_2()
    # test_wrong_state_simple()
    test_simple()