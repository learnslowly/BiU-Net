"""Write the training and evaluation configs for the array-to-WGS arm.
"""
import argparse
import os
import re

RUNID = {
    ('1KGP', 'src'): 'v3low_ftchr', ('1KGP', 'chip'): 'v3low_chipft',
    ('SGDP', 'src'): 'ft_sgdp_ftchr', ('SGDP', 'chip'): 'ft_sgdp_chipft',
}

TRAIN_HEAD = """# Array-to-WGS fine-tuning of the genome-wide merged-data model, chromosome {c}.
# Identical to configs/train_{ds}_chrft_{c}.yaml except for the masking regime:
# whole variant columns are removed for every sample at a rate drawn in
# [0.85, 0.95], and the columns that survive are ascertained above 5% minor
# allele frequency, which is the structure of a genotyping array. The test set
# this model is scored on is built the same way (scripts/masking_chip_gw.py).
"""

TEST_HEAD_CHIP = """# Array-to-WGS evaluation of the array-adapted model, chromosome {c}.
# The test set carries only the array positions; everything else is missing for
# every sample, so benchmarkAll is off and only the imputed positions are scored.
"""

TEST_HEAD_ASIS = """# Array-to-WGS evaluation of the sporadically-trained per-chromosome model,
# chromosome {c}. Same test set and same scoring as the array-adapted arm; the
# difference between the two is what matching the training regime to the test
# regime is worth.
"""

COLWISE_BLOCK = """maskMode: colwise
colwiseDynamicRange: True
colwiseRateMin: 0.85
colwiseMafMin: 0.05
"""


def set_key(text, key, value):
    """Replace a top-level scalar key, keeping any trailing comment off the line."""
    pattern = re.compile(rf"^{re.escape(key)}:.*$", re.M)
    if pattern.search(text):
        return pattern.sub(f"{key}: {value}", text, count=1)
    return text.rstrip('\n') + f"\n{key}: {value}\n"


def drop_key(text, key):
    """Remove a key the cluster's ModelConfig does not define."""
    return re.sub(rf"^{re.escape(key)}:.*\n", "", text, flags=re.M)


def strip_head(text):
    lines = text.split('\n')
    i = 0
    while i < len(lines) and lines[i].startswith('#'):
        i += 1
    return '\n'.join(lines[i:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--missing-pct', type=int, default=90,
                    help='label the array masks were written under')
    ap.add_argument('--rand-state', type=int, default=2000)
    args = ap.parse_args()
    missing = args.missing_pct / 100.0
    written = 0

    for ds in ('1KGP', 'SGDP'):
        for c in range(1, 23):
            src_train = f"configs/train_{ds}_chrft_{c}.yaml"
            src_test = f"configs/test_{ds}_chrft_{c}.yaml"
            for path in (src_train, src_test):
                if not os.path.exists(path):
                    raise FileNotFoundError(path)

            body = strip_head(open(src_train).read())
            body = set_key(body, 'runId', RUNID[(ds, 'chip')])
            body = set_key(body, 'missingRatio', '0.95')
            body = set_key(body, 'dynamicRatio', 'False')
            body = body.rstrip('\n') + '\n' + COLWISE_BLOCK
            out = f"configs/train_{ds}_chipft_{c}.yaml"
            open(out, 'w').write(TRAIN_HEAD.format(c=c, ds=ds) + body)
            written += 1

            test_body = strip_head(open(src_test).read())
            test_body = set_key(test_body, 'missing', f"[{missing}]")
            test_body = set_key(test_body, 'testRandStates', f"[{args.rand_state}]")
            test_body = set_key(test_body, 'benchmarkAll', 'False')
            # Not a field of the ModelConfig the cluster runs.
            test_body = drop_key(test_body, 'writeDosageOutputs')

            chip = set_key(test_body, 'runId', RUNID[(ds, 'chip')])
            open(f"configs/test_{ds}_chipft_{c}.yaml", 'w').write(
                TEST_HEAD_CHIP.format(c=c) + chip)
            open(f"configs/test_{ds}_chrft_chip_{c}.yaml", 'w').write(
                TEST_HEAD_ASIS.format(c=c) + test_body)
            written += 2

    print(f"wrote {written} configs")


if __name__ == '__main__':
    main()
