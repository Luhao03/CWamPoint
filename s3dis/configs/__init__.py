import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

from s3dis.configs import s3dis_s, s3dis_l, s3dis_custom

model_configs = {
    's': s3dis_s.ModelConfig(),
    'l': s3dis_l.ModelConfig(),
    'c': s3dis_custom.ModelConfig(),
}
