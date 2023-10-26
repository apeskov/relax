
import tvm
from tvm.tir.schedule import Schedule, BlockRV
from tvm import meta_schedule as ms

from tune_com import matmul_g128_KN_sym_dynm, TARGET, make_arg
from tune_2 import mds1
from tune_3 import mds2
from tune_4 import mds3

def main():
  func = matmul_g128_KN_sym_dynm
  B, M, N, K = 1, 64, 22016, 4096
  BLK, GS = 8, 128

  sch = Schedule(func, debug_mask="all")
  # mds3(m_pad=64)(sch)
  mds2(m_pad=64)(sch)
  # mds1(m_pad=64)(sch)
  
  with TARGET:
    lib = tvm.build(sch.mod)
  
  args_info = [
    ms.arg_info.TensorInfo("uint32", [K // BLK, N]),  # WGH
    ms.arg_info.TensorInfo("float16", [K // GS, N]),  # SCL
    ms.arg_info.TensorInfo("float16", [B, M, K]),     # D_IN
    ms.arg_info.TensorInfo("float16", [B, M, N]),     # D_OUT
  ]
  args = [make_arg(info) for info in args_info]

  score_s = lib.time_evaluator(lib.entry_name, dev=tvm.cuda(0), number=2000, repeat=1)(*args).mean
  print(f"SCORE M: {M} TIME: {score_s*1e6} us",)


if __name__ == "__main__":
  main()
