"""모델과 문제 데이터를 미리 받는다. 한 번만 돌리면 된다."""
from huggingface_hub import snapshot_download

from . import common as C

if __name__ == "__main__":
    path = snapshot_download(C.MODEL_ID, revision=C.MODEL_REV)
    print("model snapshot:", path)
    for ds in ("gsm8k", "math500"):
        print(ds, len(C.load_problems(ds)), "problems")
    tok = C.load_tokenizer()
    print("newline token ids:", len(C.newline_ids(tok)))
