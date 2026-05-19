"""
inference.py
------------
LLM inference for the Marathi Math Tutor.

Priority chain:
  1. Fine-tuned LoRA model  (if marathi_tutor_lora/ exists)
  2. Dataset retrieval       (fast, always available)

Public API (used by server.py / app.py):
    generate_response(user_input)   → Marathi string
    explain(a, b, operation)        → Marathi explanation
    story(a, b, operation)          → Marathi story
    pythagoras_explain()            → Marathi Pythagoras explanation
    pythagoras_story()              → Marathi Pythagoras story
    pythagoras_example()            → Marathi worked example
"""

import os, json, re, random, difflib
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
_HERE       = os.path.dirname(os.path.abspath(__file__))
LORA_DIR    = os.path.join(_HERE, "marathi_tutor_lora")
META_FILE   = os.path.join(LORA_DIR, "tutor_meta.json")
DATASET_DIR = os.path.join(_HERE, "..", "..", "dataset")
MATH_JSONL  = os.path.join(DATASET_DIR, "marathi_math_dataset.jsonl")
PYTH_JSONL  = os.path.join(DATASET_DIR, "pythagoras_dataset.jsonl")

SYSTEM_PROMPT = (
    "तुम्ही एक मराठी गणित शिक्षक आहात जे दृष्टिहीन विद्यार्थ्यांना "
    "अंकगणित आणि पायथागोरस प्रमेय शिकवता. "
    "नेहमी मराठीत उत्तर द्या. स्पष्ट, सोप्या भाषेत समजावून सांगा."
)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset fallback (always available, no GPU needed)
# ══════════════════════════════════════════════════════════════════════════════

class DatasetFallback:
    """Fast keyword + fuzzy retrieval from JSONL datasets."""

    def __init__(self):
        self._math: list[dict] = []
        self._pyth: list[dict] = []
        self._load()

    def _load(self):
        def read(path):
            rows = []
            if not os.path.exists(path):
                return rows
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
            return rows

        self._math = read(MATH_JSONL)
        self._pyth = read(PYTH_JSONL)
        total = len(self._math) + len(self._pyth)
        print(f"[FALLBACK] Loaded {total} dataset entries "
              f"({len(self._math)} math + {len(self._pyth)} pythagoras)")

    # ── Pythagoras responses ───────────────────────────────────────────────

    def pyth_response(self, mode: str) -> str:
        """
        mode: 'explain' | 'story' | 'example'
        Returns a rich multi-line response from the Pythagoras dataset.
        """
        key_map = {
            "explain": "पायथागोरस समजाव",
            "story":   "गोष्ट वापरून पायथागोरस समजाव",
            "example": "उदाहरण दे",
        }
        key = key_map.get(mode, "पायथागोरस समजाव")
        candidates = [d for d in self._pyth if d.get("instruction") == key]
        if candidates:
            return random.choice(candidates).get("response", "")
        if self._pyth:
            return random.choice(self._pyth).get("response", "")
        return "पायथागोरस प्रमेयानुसार a² + b² = c² असते."

    # ── Math responses ─────────────────────────────────────────────────────

    def math_response(self, instruction: str) -> str:
        """Retrieve best math response using exact → template → fuzzy matching."""
        if not self._math:
            return "माफ करा, उत्तर सापडले नाही."

        # 1. Exact match
        for d in self._math:
            if d.get("instruction") == instruction:
                return d.get("response", "")

        # 2. Template substitution (same op + mode, substitute numbers)
        query_nums = re.findall(r'\d+', instruction)
        op_char = next((c for c in ["+", "-", "×", "÷"] if c in instruction), None)
        mode = "गोष्टीतून" if "गोष्टीतून" in instruction else "समजाव"

        if op_char and len(query_nums) >= 2:
            a, b = int(query_nums[0]), int(query_nums[1])
            candidates = [
                d for d in self._math
                if op_char in d.get("instruction", "")
                and mode in d.get("instruction", "")
            ]
            if candidates:
                tmpl = random.choice(candidates)
                resp = tmpl["response"]
                orig_nums = re.findall(r'\d+', tmpl["instruction"])
                if len(orig_nums) >= 2:
                    oa, ob = int(orig_nums[0]), int(orig_nums[1])
                    op_fn = {"+": lambda x,y: x+y, "-": lambda x,y: x-y,
                             "×": lambda x,y: x*y, "÷": lambda x,y: x//y if y else 0}
                    result  = op_fn[op_char](a, b)
                    o_result = op_fn[op_char](oa, ob)
                    for old, new in sorted(
                        [(str(o_result), str(result)), (str(oa), str(a)), (str(ob), str(b))],
                        key=lambda x: len(x[0]), reverse=True
                    ):
                        resp = resp.replace(old, new)
                return resp

        # 3. Fuzzy match
        instructions = [d.get("instruction", "") for d in self._math]
        matches = difflib.get_close_matches(instruction, instructions, n=1, cutoff=0.4)
        if matches:
            for d in self._math:
                if d.get("instruction") == matches[0]:
                    return d.get("response", "")

        return random.choice(self._math).get("response", "")


# ══════════════════════════════════════════════════════════════════════════════
# LLM wrapper
# ══════════════════════════════════════════════════════════════════════════════

class MarathiTutorLLM:
    """
    Wraps the fine-tuned LoRA model.
    Falls back to DatasetFallback if model is not trained yet.
    """

    def __init__(self):
        self.pipe     = None
        self.meta     = {}
        self.fallback = DatasetFallback()
        self._load_model()

    def _load_model(self):
        if not os.path.isdir(LORA_DIR):
            print("[LLM] No fine-tuned model found. Using dataset fallback.")
            print(f"      Train with: python backend/llm_model/finetune.py")
            return

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from peft import PeftModel

            # Load saved metadata
            if os.path.exists(META_FILE):
                with open(META_FILE, encoding="utf-8") as f:
                    self.meta = json.load(f)

            base_model_id = self.meta.get("base_model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            max_new_tokens = self.meta.get("max_new_tokens", 300)

            print(f"[LLM] Loading fine-tuned model from {LORA_DIR} ...")
            tokenizer = AutoTokenizer.from_pretrained(LORA_DIR)

            use_gpu = torch.cuda.is_available()
            base = AutoModelForCausalLM.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                low_cpu_mem_usage=True,
                device_map="auto" if use_gpu else None,
            )
            model = PeftModel.from_pretrained(base, LORA_DIR)
            model.eval()

            self.pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.eos_token_id,
            )
            print("[LLM] Model loaded successfully ✓")

        except Exception as e:
            print(f"[LLM] Could not load model: {e}")
            print("[LLM] Falling back to dataset retrieval.")
            self.pipe = None

    def _build_prompt(self, instruction: str, inp: str = "") -> str:
        user_part = f"{instruction}\n{inp}".strip() if inp else instruction
        return (
            f"<|system|>\n{SYSTEM_PROMPT}</s>\n"
            f"<|user|>\n{user_part}</s>\n"
            f"<|assistant|>\n"
        )

    def _extract_response(self, generated: str, prompt: str) -> str:
        """Strip the prompt prefix and clean up the generated text."""
        if "<|assistant|>" in generated:
            text = generated.split("<|assistant|>")[-1]
        elif prompt in generated:
            text = generated[len(prompt):]
        else:
            text = generated
        # Stop at next turn marker
        for stop in ["</s>", "<|user|>", "<|system|>"]:
            if stop in text:
                text = text[:text.index(stop)]
        return text.strip()

    def generate(self, instruction: str, inp: str = "") -> str:
        """Generate a Marathi response. Falls back to dataset if model unavailable."""
        if self.pipe is None:
            return self.fallback.math_response(instruction)

        prompt = self._build_prompt(instruction, inp)
        try:
            out = self.pipe(prompt)[0]["generated_text"]
            return self._extract_response(out, prompt)
        except Exception as e:
            print(f"[LLM] Generation error: {e}")
            return self.fallback.math_response(instruction)


# ── Singleton ──────────────────────────────────────────────────────────────────
_tutor: MarathiTutorLLM | None = None

def _get_tutor() -> MarathiTutorLLM:
    global _tutor
    if _tutor is None:
        _tutor = MarathiTutorLLM()
    return _tutor


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def generate_response(user_input: str) -> str:
    """
    Main entry point.
    Accepts any Marathi instruction and returns a Marathi response.
    """
    return _get_tutor().generate(user_input)


def explain(a: int, b: int, operation: str) -> str:
    """Step-by-step explanation of an arithmetic operation."""
    op_map = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}
    symbol = op_map.get(operation, "+")
    instruction = f"{a}{symbol}{b} समजाव"
    t = _get_tutor()
    if t.pipe:
        return t.generate(instruction)
    return t.fallback.math_response(instruction)


def story(a: int, b: int, operation: str) -> str:
    """Story-based explanation of an arithmetic operation."""
    op_map = {"add": "+", "sub": "-", "mul": "×", "div": "÷"}
    symbol = op_map.get(operation, "+")
    instruction = f"{a}{symbol}{b} गोष्टीतून समजाव"
    t = _get_tutor()
    if t.pipe:
        return t.generate(instruction)
    return t.fallback.math_response(instruction)


def pythagoras_explain() -> str:
    """Detailed explanation of the Pythagorean theorem."""
    t = _get_tutor()
    if t.pipe:
        return t.generate("पायथागोरस प्रमेय समजाव. सूत्र, उदाहरण आणि उपयोग सांगा.")
    return t.fallback.pyth_response("explain")


def pythagoras_story() -> str:
    """Story-based explanation of the Pythagorean theorem."""
    t = _get_tutor()
    if t.pipe:
        return t.generate("पायथागोरस प्रमेय गोष्टीतून समजाव.")
    return t.fallback.pyth_response("story")


def pythagoras_example() -> str:
    """Step-by-step worked example using a Pythagorean triple."""
    t = _get_tutor()
    if t.pipe:
        return t.generate("पायथागोरस प्रमेयाचे एक उदाहरण पायरी पायरी सोडव.")
    return t.fallback.pyth_response("example")


# ── CLI test ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test Marathi Tutor LLM inference")
    parser.add_argument("--prompt", type=str, default=None)
    args = parser.parse_args()

    if args.prompt:
        print(generate_response(args.prompt))
    else:
        print("=== Inference Test ===\n")
        tests = [
            ("explain",  lambda: explain(5, 3, "add")),
            ("story",    lambda: story(10, 4, "sub")),
            ("pyth_exp", lambda: pythagoras_explain()),
            ("pyth_st",  lambda: pythagoras_story()),
            ("pyth_ex",  lambda: pythagoras_example()),
        ]
        for name, fn in tests:
            print(f"[{name}]")
            print(fn()[:200])
            print()
