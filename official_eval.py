import os

EVAL_DIR = 'eval'


def official_f1():
    # Run the perl script
    try:
        cmd = "perl {0}/semeval2010_task8_scorer-v1.2.pl {0}/proposed_answers.txt {0}/answer_keys.txt > {0}/result.txt".format(EVAL_DIR)
        os.system(cmd)
    except:
        raise Exception("perl is not installed or proposed_answers.txt is missing")

    with open(os.path.join(EVAL_DIR, 'result.txt'), 'r', encoding='utf-8') as f:
        macro_result = list(f)[-1]
        macro_result = macro_result.split(":")[1].replace(">>>", "").strip()
        macro_result = macro_result.split("=")[1].strip().replace("%", "")
        macro_result = float(macro_result) / 100

    return macro_result


if __name__ == "__main__":
    print("macro-averaged F1 = {}%".format(official_f1() * 100))
