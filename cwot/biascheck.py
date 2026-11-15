"""실험 4b의 로짓 처리기가 실제로 적용되는지 확인한다.

  python -m cwot.biascheck
1) </think> 로짓에 +100을 더하면 첫 토큰이 </think>여야 한다.
2) 더한 값이 0, 2.5, 100일 때 처리기가 불린 횟수와, 처리기 앞뒤의 </think> 로짓 차이를 찍는다.
"""
import torch
from transformers import LogitsProcessorList

from . import common as C
from .exp4b_bias import CloseBias, Presence


class Spy(CloseBias):
    calls = 0
    seen = None

    def __call__(self, input_ids, scores):
        out = super().__call__(input_ids, scores)
        Spy.calls += 1
        if Spy.seen is None:
            Spy.seen = (float(scores[0, self.close_id]), float(out[0, self.close_id]), float(scores[0].max()), int(scores[0].argmax()))
        return out


def main():
    tok = C.load_tokenizer()
    model = C.load_model()
    close_id = C.enc(tok, C.THINK_CLOSE)[0]
    probs = C.load_problems("gsm8k")
    ids = C.enc(tok, C.prompt_text(tok, probs["gsm8k_test_0"]["problem"]))
    n0 = len(ids)
    x = torch.tensor([ids], device=model.device)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    import transformers
    print("transformers", transformers.__version__, "| close_id", close_id, "| n_prompt", n0)
    for b in (0.0, 2.5, 100.0):
        Spy.calls, Spy.seen = 0, None
        torch.manual_seed(0)
        with torch.no_grad():
            out = model.generate(input_ids=x, attention_mask=torch.ones_like(x), do_sample=True, temperature=0.6, top_p=0.95, top_k=20,
                                 repetition_penalty=1.0, max_new_tokens=6, pad_token_id=pad,
                                 logits_processor=LogitsProcessorList([Presence(n0, 1.5), Spy(n0, close_id, b)]))
        gen = out[0, n0:].tolist()
        print(f"bias {b}: calls={Spy.calls} first-step (close logit before, after, max logit, argmax)={Spy.seen} gen={gen} text={tok.decode(gen)!r}")
    print("first token is </think> with +100:", gen[0] == close_id)
    print("BIASCHECK DONE")


if __name__ == "__main__":
    main()
