import tvm
import tvm.meta_schedule as ms
from tvm import te
from tvm._ffi import get_global_func

import tvm.script.tir as T
from tvm.tir.schedule import BlockRV, Schedule

import tvm.tir.tensor_intrin.cuda


def gen_simple_mm(M, N, K, dim_dtype="int32"):
    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        M, N, K = (T.int64(v) for v in (M, N, K))

    a = te.placeholder((M, K), name="A", dtype="float16")
    wgh = te.placeholder((K, N), name="WGH", dtype="float16")

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (M, N),
        lambda vm, vn: te.sum(a[vm, k] * wgh[k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, wgh, x])


def gen_simple_bmm(B, M, N, K, dim_dtype="int32"):
    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        B, M, N, K = (T.int64(v) for v in (B, M, N, K))

    a = te.placeholder((B, M, K), name="A", dtype="float16")
    wgh = te.placeholder((B, K, N), name="WGH", dtype="float16")

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (B, M, N),
        lambda vb, vm, vn: te.sum(a[vb, vm, k] * wgh[vb, k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, wgh, x])

def gen_simple_qmm(M, N, K, GS, dim_dtype="int32"):
    assert K % GS == 0
    G = K // GS

    BLK = 8

    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        M, N, K, GS, G, BLK = (T.int64(v) for v in (M, N, K, GS, G, BLK))


    # Groupt quantization
    a = te.placeholder((M, K), name="A", dtype="float16")
    qwgh = te.placeholder((K // BLK, N), name="QWGH", dtype="uint32")
    scl = te.placeholder((G, N), name="SCL", dtype="float16")

    # Decode weights
    wgh = te.compute(
        (K, N),
        lambda vk, vn: 
            ((qwgh[vk // BLK, vn] >> (vk % BLK * 4) & 0xF) - 7).astype("float16") * 
            scl[vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (M, N),
        lambda vm, vn: te.sum(a[vm, k] * wgh[k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])


def gen_simple_qbmm(B, M, N, K, GS, dim_dtype="int32"):
    assert K % GS == 0
    G = K // GS

    BLK = 8

    assert dim_dtype in ["int64", "int32"]
    if dim_dtype == "int64":
        B, M, N, K, GS, G, BLK = (T.int64(v) for v in (B, M, N, K, GS, G, BLK))


    # Groupt quantization
    a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((B, K // BLK, N), name="QWGH", dtype="uint32")
    scl = te.placeholder((B, G, N), name="SCL", dtype="float16")

    # Decode weights
    wgh = te.compute(
        (B, K, N),
        lambda vb, vk, vn: 
            ((qwgh[vb, vk // BLK, vn] >> (vk % BLK * 4) & 0xF) - 7).astype("float16") * 
            scl[vb, vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (B, M, N),
        lambda vb, vm, vn: te.sum(a[vb, vm, k] * wgh[vb, k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])


def gen_simple_qbrgemm_64(B, M, N, K, GS):
    assert K % GS == 0
    G = K // GS

    # convert dims to T.int64
    B, M, N, K, GS, G = (T.int64(v) for v in (B, M, N, K, GS, G))

    # Groupt quantization
    a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((B, K // T.int64(8), N), name="QWGH", dtype="uint32")
    scl = te.placeholder((B, G, N), name="SCL", dtype="float16")
          
    # Decode weights
    wgh = te.compute(
        (B, K, N),
        lambda vb, vk, vn: 
            ((qwgh[vb, vk // T.int64(8), vn] >> (vk % T.int64(8) * 4) & 0xF) - 7).astype("float16") * 
            scl[vb, vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (B, M, N),
        lambda vb, vm, vn: te.sum(a[vb, vm, k] * wgh[vb, k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])


def gen_simple_qbrgemm(B, M, N, K, GS):
    assert K % GS == 0
    G = K // GS

    # Groupt quantization
    a = te.placeholder((B, M, K), name="A", dtype="float16")
    qwgh = te.placeholder((B, K//8, N), name="QWGH", dtype="uint32")
    scl = te.placeholder((B, G, N), name="SCL", dtype="float16")
          
    # Decode weights
    wgh = te.compute(
        (B, K, N),
        lambda vb, vk, vn: 
            ((qwgh[vb, vk // 8, vn] >> (vk % 8 * 4) & 0xF) - 7).astype("float16") * 
            scl[vb, vk // GS, vn],
        name="decode_wgh",
    )

    # Matmul
    k = te.reduce_axis((0, K), name="k")
    x = te.compute(
        (B, M, N),
        lambda vb, vm, vn: te.sum(a[vb, vm, k] * wgh[vb, k, vn], axis=k),
        name="matmul"
    )

    return tvm.te.create_prim_func([a, qwgh, scl, x])


def main():
    B, M, N, K = 16, 1, 22016, 4096
    GS = 128
    dim_dtype = "int32"
    # dim_dtype = "int64"
    
    # funcs = {f"simple_qbrgemm_{B}_{M}_{N}_{K//B}": gen_simple_qbrgemm_64(B, M, N, K//B, GS=128)}
    # funcs = {f"simple_mm_{dim_dtype}_{M}_{N}_{K}": gen_simple_mm(M, N, K, dim_dtype)}
    # funcs = {f"simple_bmm_{dim_dtype}_{B}_{M}_{N}_{K}": gen_simple_bmm(B, M, N, K, dim_dtype)}  // In case of mistake M, N, K, GS it works 
    funcs = {f"simple_qmm_{dim_dtype}_{M}_{N}_{K}_{GS}": gen_simple_qmm(M, N, K, GS, dim_dtype)}
    mod = tvm.IRModule(funcs)

    target = tvm.target.Target("nvidia/nvidia-a10g")

    ms.tir_integration.tune_tir(
        mod=mod,
        target=target,
        work_dir=f"__tmp/tune_simple_qmm_{dim_dtype}",
        max_trials_global=100500,
        max_trials_per_task=4,
        num_trials_per_iter=2,
        cost_model="random",
        # cost_model="xgb",
    )


@ms.utils.derived_object
class RFactorScheduleRule(ms.schedule_rule.PyScheduleRule):
    def __init__(self, mlt_rule) -> None:
        super().__init__()
        self._mlt_rule = mlt_rule

    def _initialize_with_tune_context(self, context) -> None:
        pass
    
    def accepted_by_mlt_rule(self, sch, block):
        """Check if provided block is gemm
        As a temporal solution we can check if mlt_rule can be applyed to it.
        """
        res_schs = self._mlt_rule.apply(sch, block)
        return len(res_schs) > 1 or res_schs[0] != sch        


    def apply(self, sch: Schedule, block: BlockRV):
        if not self.accepted_by_mlt_rule(sch, block):
            return [sch]

        RF = 4  # make it tunable

        new_sch = sch.copy()
        # Apply "rfactor"
        loops = new_sch.get_loops(block)
        lk = loops[-1]
        lko, lki = new_sch.split(lk, factors=[RF, None])
        new_sch.reorder(lko, *loops[0:-1], lki)
        main_rf_block = new_sch.rfactor(lko, factor_axis=0)  # TODO: does factor_axis=0 affect performance?
        final_rf_block, = new_sch.get_consumers(main_rf_block)
        
        # Apply original multi level tiling rule for main reduction block
        res_schs = self._mlt_rule.apply(new_sch, main_rf_block)

        # Schedule final reduction block
        def merge_final_reduction(sch):
            print(sch.get(final_rf_block))
        
        res_schs = [merge_final_reduction(s, ) for s in res_schs]

        return res_schs

    def clone(self) -> ms.schedule_rule.ScheduleRule:
        new_mlt_rule = self._mlt_rule.clone()
        return RFactorScheduleRule(new_mlt_rule)


def main_rf_custom():
    M, N, K = 1, 22016, 4096
    GS = 128
    
    funcs = {f"simple_qmm_{M}_{N}_{K}_{GS}": gen_simple_qmm(M, N, K, GS, dim_dtype="int32")}
    mod = tvm.IRModule(funcs)

    target = tvm.target.Target("nvidia/nvidia-a10g")
    rules_kind = "cuda-tensorcore"

    rules = ms.schedule_rule.schedule_rule.create(rules_kind)
    new_rules = rules[0:1]
    new_rules += [RFactorScheduleRule(r) for r in rules[1:2]]  # Add rfactor injection on top of original 
    new_rules += rules[3:]

    ms.tir_integration.tune_tir(
        mod=mod,
        target=target,
        work_dir=f"__tmp/tune_simple_qmm_xxx",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.PostOrderApply(
            sch_rules=new_rules,
            postprocs=rules_kind,
            mutator_probs=rules_kind,
        )
    )


def manual_read_db(db_path):
    workload_from_json = get_global_func("meta_schedule.WorkloadFromJSON")
    tuning_record_from_json = get_global_func("meta_schedule.TuningRecordFromJSON")
    json_loads = get_global_func("meta_schedule.JSONLoads")

    wklds = []
    with open(f"{db_path}/database_workload.json", "r") as f:
        for line in f.readlines():
            json_line = json_loads(line) 
            wkld = workload_from_json(json_line)
            wklds.append(wkld)

    recs = []
    with open(f"{db_path}/database_tuning_record.json", "r") as f:
        for line in f.readlines():
            json_line = json_loads(line)
            idx = int(json_line[0])
            rec_json = json_line[1]
            
            wkld = wklds[idx]
            sch = tvm.tir.Schedule(wkld.mod)
            
            tvm.tir.schedule.Trace.apply_json_to_schedule(rec_json[0], sch)

            rec = tuning_record_from_json(rec_json, wklds[idx])
            recs.append(rec)
    
    return recs


def read_db():
    db_path = "__tmp/tune_simple_qmm_int32"
    manual_read_db(db_path)
    

if __name__ == "__main__":
    # main()
    main_rf_custom()
    # read_db()
