import tvm
from tvm import meta_schedule as ms
from tvm.tir import Schedule
from tvm.tir.schedule import BlockRV

from tune_com import matmul_g128_KN_sym_dynm, TARGET, DEV

"""
StD Tune

This file implement dynamic tuning version one.
Use standard ms.ScheduleRules and wrapper with additional rfactors.
Finaly convert results to dynamic version.
"""

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

        return res_schs

    def clone(self) -> ms.schedule_rule.ScheduleRule:
        new_mlt_rule = self._mlt_rule.clone()
        return RFactorScheduleRule(new_mlt_rule)


def main_rf_custom():    
    mod = tvm.IRModule({"matmul_g128_KN_sym_dynm": matmul_g128_KN_sym_dynm})

    ms_rule_kind = "cuda-tensorcore"

    rules = ms.schedule_rule.schedule_rule.create(ms_rule_kind)
    rules[1:2] = [RFactorScheduleRule(r) for r in rules[1:2]]  # Add rfactor injection on top of original 

    ms.tir_integration.tune_tir(
        mod=mod,
        target=TARGET,
        work_dir=f"___tmp/tune_simple_qmm_rf",
        max_trials_global=100500,
        max_trials_per_task=4096,
        num_trials_per_iter=32,
        cost_model="random",
        # cost_model="xgb",
        space=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=ms_rule_kind,
            mutator_probs=ms_rule_kind,
        )
    )
