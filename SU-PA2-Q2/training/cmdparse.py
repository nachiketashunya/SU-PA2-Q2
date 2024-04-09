import speechbrain as sb
import sys
from hyperpyyaml import load_hyperpyyaml

hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

# print(hparams_file)
print(run_opts)
print(overrides)
print(type(overrides))
