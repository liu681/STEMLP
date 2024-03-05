from .arch_zoo.stemlp_arch import STEMLP
from .arch_zoo.stid_arch import STID
from .arch_zoo.gwnet_arch import GraphWaveNet
from .arch_zoo.stgcn_arch import STGCN
from .arch_zoo.mtgnn_arch import MTGNN
from .arch_zoo.stnorm_arch import STNorm
from .arch_zoo.stemgnn_arch import StemGNN
from .arch_zoo.gts_arch import GTS
from .arch_zoo.dgcrn_arch import DGCRN
from .arch_zoo.hi_arch import HINetwork


__all__ = ["STEMLP","STID", "GraphWaveNet",  "STGCN", "MTGNN",
           "STNorm",  "StemGNN",
           "GTS", "DGCRN",
           "HINetwork"]
