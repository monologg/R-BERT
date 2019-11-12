import os
import subprocess

EVAL_DIR = 'eval'

cmd = "perl {0}/semeval2010_task8_scorer-v1.2.pl {0}/proposed_answers.txt {0}/answer_keys.txt > {0}/result.txt".format(EVAL_DIR)

os.system(cmd)


with open(os.path.join(EVAL_DIR, 'result.txt'), 'r', encoding='utf-8') as f:
    macro_result = list(f)[-1]
    macro_result = macro_result.replace("<<<", "").replace(">>>", "").strip()
    print(macro_result)