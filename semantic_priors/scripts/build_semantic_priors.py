import argparse
import json
import os
import re
import urllib.request
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import yaml
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


SLOTS = ["type", "composition", "morphology", "texture", "context", "condition"]


@dataclass
class ClassSpec:
    class_id: int
    name: str
    alias: List[str]
    definition: str


class SemanticPriorBuilder:
    """Hierarchical semantic-prior constructor for Chapter 4.

    Pipeline:
    1) hierarchical text generation (coarse + fine src/tgt)
    2) K-sampling and rule-based aggregation
    3) text embedding + l2 normalization
    4) consistency-based confidence estimation
    5) export prototype banks and merged bank
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.dataset = cfg["dataset"]
        self.classes = self._parse_classes(cfg["classes"])
        self.scene_info = cfg["scene_info"]
        self.k = int(cfg.get("generation", {}).get("k", 5))
        self.rho = float(cfg.get("generation", {}).get("rho", 0.6))
        self.max_sentences = int(cfg.get("generation", {}).get("max_sentences", 4))
        self.min_weight = float(cfg.get("confidence", {}).get("eps", 0.05))
        self.encoding_dim = int(cfg.get("encoding", {}).get("dim", 256))
        self.output_dir = cfg.get("output_dir", f"semantic_priors/{self.dataset}")
        self.merge_mode = cfg.get("output", {}).get("merge_mode", "weighted_mean")
        self.generation_backend = cfg.get("generation", {}).get("backend", "template")
        self.openai_cfg = cfg.get("openai", {})

    @staticmethod
    def _parse_classes(raw: List[Dict]) -> List[ClassSpec]:
        classes = []
        for item in raw:
            classes.append(
                ClassSpec(
                    class_id=int(item["id"]),
                    name=item["name"],
                    alias=item.get("alias", []),
                    definition=item.get("definition", "unknown"),
                )
            )
        classes = sorted(classes, key=lambda x: x.class_id)
        return classes


    def _call_openai_compatible(self, prompt: str) -> str:
        api_key_env = self.openai_cfg.get("api_key_env", "OPENAI_API_KEY")
        api_key = os.getenv(api_key_env, "")
        if not api_key:
            raise RuntimeError(f"Missing API key env: {api_key_env}")

        base_url = self.openai_cfg.get("base_url", "https://api.openai.com")
        model = self.openai_cfg.get("model", "gpt-4o-mini")
        temperature = float(self.openai_cfg.get("temperature", 0.2))
        max_tokens = int(self.openai_cfg.get("max_tokens", 260))

        url = base_url.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a remote-sensing semantic assistant."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))
        return obj["choices"][0]["message"]["content"].strip()

    def _build_coarse_prompt(self, cls: ClassSpec, seed: int) -> str:
        alias = ", ".join(cls.alias) if cls.alias else "none"
        return (
            f"Class name: {cls.name}\n"
            f"Alias: {alias}\n"
            f"Definition: {cls.definition}\n"
            "Task: generate concise coarse-grained, scene-invariant remote-sensing semantics. "
            "Must include slots exactly: Type, Composition, Morphology, Texture, Context, Condition. "
            "Use objective statements only and keep <= 6 sentences. "
            f"Variant marker: {seed}."
        )

    def _build_fine_prompt(self, cls: ClassSpec, domain: str, seed: int) -> str:
        alias = ", ".join(cls.alias) if cls.alias else "none"
        scene_txt = self._scene_to_text(domain)
        return (
            f"Class name: {cls.name}\n"
            f"Alias: {alias}\n"
            f"Definition: {cls.definition}\n"
            f"Scene info ({domain}): {scene_txt}\n"
            "Task: generate concise fine-grained, scene-aware remote-sensing semantics. "
            "Must include slots exactly: Type, Composition, Morphology, Texture, Context, Condition. "
            "Do not fabricate unknown fields and keep <= 6 sentences. "
            f"Variant marker: {seed}."
        )

    def _scene_to_text(self, domain: str) -> str:
        info = self.scene_info.get(domain, {})
        ordered_keys = ["sensor", "season", "region", "resolution", "illumination"]
        parts = [f"{k}={info.get(k, 'unknown')}" for k in ordered_keys]
        return "; ".join(parts)

    def _generate_coarse_once(self, cls: ClassSpec, seed: int) -> str:
        if self.generation_backend == "openai":
            prompt = self._build_coarse_prompt(cls, seed)
            return self._call_openai_compatible(prompt)
        alias = ", ".join(cls.alias) if cls.alias else "none"
        return (
            f"Type: {cls.name} is a land-cover category in hyperspectral remote sensing. "
            f"Composition: {cls.definition}. Known aliases: {alias}. "
            f"Morphology: typically shows stable spatial organization and class-specific structural layout. "
            f"Texture: presents relatively consistent spectral-spatial texture under cross-scene observation. "
            f"Context: appears with meaningful neighborhood relations to surrounding classes. "
            f"Condition: coarse description focuses on scene-invariant semantic properties. "
            f"Variant marker: v{seed}."
        )

    def _generate_fine_once(self, cls: ClassSpec, domain: str, seed: int) -> str:
        if self.generation_backend == "openai":
            prompt = self._build_fine_prompt(cls, domain, seed)
            return self._call_openai_compatible(prompt)
        scene_txt = self._scene_to_text(domain)
        return (
            f"Type: {cls.name} under {domain} domain condition. "
            f"Composition: follows class definition ({cls.definition}) with domain-aware observation details. "
            f"Morphology: boundary visibility and shape continuity may vary with scene setting ({scene_txt}). "
            f"Texture: texture density, local contrast, and spectral response can change across conditions. "
            f"Context: neighboring category interactions can shift under domain context ({domain}). "
            f"Condition: scene-aware factors include {scene_txt}; unknown fields remain unknown. "
            f"Variant marker: v{seed}."
        )

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def _extract_units(self, text: str) -> Dict[str, str]:
        units = {}
        for slot in SLOTS:
            m = re.search(rf"{slot}:\s*(.*?)(?:\.\s|$)", text, flags=re.IGNORECASE)
            if m:
                units[slot] = self._normalize_text(m.group(1))
        return units

    def _aggregate_candidates(self, texts: List[str]) -> str:
        unit_counts: Dict[Tuple[str, str], int] = {}
        for t in texts:
            units = self._extract_units(t)
            for slot, val in units.items():
                key = (slot, val)
                unit_counts[key] = unit_counts.get(key, 0) + 1

        kept = {slot: [] for slot in SLOTS}
        threshold = max(1, int(np.ceil(self.rho * len(texts))))
        for (slot, val), cnt in unit_counts.items():
            if cnt >= threshold:
                kept[slot].append((val, cnt))

        composed = []
        for slot in SLOTS:
            if kept[slot]:
                kept[slot].sort(key=lambda x: (-x[1], x[0]))
                composed.append(f"{slot}: {kept[slot][0][0]}")

        if not composed:
            composed = [self._normalize_text(texts[0])]

        truncated = ". ".join(composed[: self.max_sentences])
        if not truncated.endswith("."):
            truncated += "."
        return truncated

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        tfidf = TfidfVectorizer(max_features=max(1024, self.encoding_dim * 8), ngram_range=(1, 2))
        x = tfidf.fit_transform(texts)

        if x.shape[1] > self.encoding_dim:
            svd_dim = min(self.encoding_dim, x.shape[0] - 1, x.shape[1] - 1)
            if svd_dim >= 2:
                svd = TruncatedSVD(n_components=svd_dim, random_state=0)
                z = svd.fit_transform(x)
            else:
                z = x.toarray()
        else:
            z = x.toarray()

        # pad/crop to target dim for stable downstream shape
        if z.shape[1] < self.encoding_dim:
            pad = np.zeros((z.shape[0], self.encoding_dim - z.shape[1]), dtype=np.float32)
            z = np.concatenate([z.astype(np.float32), pad], axis=1)
        else:
            z = z[:, : self.encoding_dim].astype(np.float32)

        norm = np.linalg.norm(z, axis=1, keepdims=True)
        z = z / np.clip(norm, 1e-12, None)
        return z

    @staticmethod
    def _mean_pairwise_cosine(embs: np.ndarray) -> float:
        if embs.shape[0] <= 1:
            return 1.0
        sim_sum, cnt = 0.0, 0
        for i in range(embs.shape[0]):
            for j in range(i + 1, embs.shape[0]):
                sim_sum += float(np.dot(embs[i], embs[j]))
                cnt += 1
        return sim_sum / max(cnt, 1)

    def build(self):
        os.makedirs(self.output_dir, exist_ok=True)

        final_texts = {"coarse": [], "fine_src": [], "fine_tgt": []}
        candidate_texts = {"coarse": {}, "fine_src": {}, "fine_tgt": {}}

        for cls in self.classes:
            coarse_candidates = [self._generate_coarse_once(cls, seed=k) for k in range(self.k)]
            src_candidates = [self._generate_fine_once(cls, "src", seed=k) for k in range(self.k)]
            tgt_candidates = [self._generate_fine_once(cls, "tgt", seed=k) for k in range(self.k)]

            candidate_texts["coarse"][cls.class_id] = coarse_candidates
            candidate_texts["fine_src"][cls.class_id] = src_candidates
            candidate_texts["fine_tgt"][cls.class_id] = tgt_candidates

            final_texts["coarse"].append(self._aggregate_candidates(coarse_candidates))
            final_texts["fine_src"].append(self._aggregate_candidates(src_candidates))
            final_texts["fine_tgt"].append(self._aggregate_candidates(tgt_candidates))

        # embed final texts
        all_final = final_texts["coarse"] + final_texts["fine_src"] + final_texts["fine_tgt"]
        all_emb = self._embed_texts(all_final)
        c = len(self.classes)
        p_c = all_emb[:c]
        p_f_src = all_emb[c : 2 * c]
        p_f_tgt = all_emb[2 * c : 3 * c]

        # confidence from candidate consistency
        confidence = {"coarse": [], "fine_src": [], "fine_tgt": []}
        for idx, cls in enumerate(self.classes):
            for key in ["coarse", "fine_src", "fine_tgt"]:
                cand = candidate_texts[key][cls.class_id]
                emb = self._embed_texts(cand)
                r = self._mean_pairwise_cosine(emb)
                w = np.clip((r + 1.0) / 2.0, self.min_weight, 1.0)
                confidence[key].append(float(w))

        w_c = np.array(confidence["coarse"], dtype=np.float32).reshape(-1, 1)
        w_src = np.array(confidence["fine_src"], dtype=np.float32).reshape(-1, 1)
        w_tgt = np.array(confidence["fine_tgt"], dtype=np.float32).reshape(-1, 1)

        if self.merge_mode == "weighted_mean":
            merged = (w_c * p_c + w_src * p_f_src + w_tgt * p_f_tgt) / np.clip(w_c + w_src + w_tgt, 1e-6, None)
        elif self.merge_mode == "coarse_only":
            merged = p_c
        elif self.merge_mode == "fine_src_only":
            merged = p_f_src
        elif self.merge_mode == "fine_tgt_only":
            merged = p_f_tgt
        else:
            raise ValueError(f"Unsupported merge_mode: {self.merge_mode}")

        merged = merged / np.clip(np.linalg.norm(merged, axis=1, keepdims=True), 1e-12, None)

        np.save(os.path.join(self.output_dir, "semantic_bank_coarse.npy"), p_c)
        np.save(os.path.join(self.output_dir, "semantic_bank_fine_src.npy"), p_f_src)
        np.save(os.path.join(self.output_dir, "semantic_bank_fine_tgt.npy"), p_f_tgt)
        np.save(os.path.join(self.output_dir, "semantic_bank_combined.npy"), merged)
        np.save(os.path.join(self.output_dir, "semantic_weights_coarse.npy"), w_c.squeeze(1))
        np.save(os.path.join(self.output_dir, "semantic_weights_fine_src.npy"), w_src.squeeze(1))
        np.save(os.path.join(self.output_dir, "semantic_weights_fine_tgt.npy"), w_tgt.squeeze(1))

        metadata = {
            "dataset": self.dataset,
            "num_classes": len(self.classes),
            "encoding_dim": self.encoding_dim,
            "k": self.k,
            "rho": self.rho,
            "merge_mode": self.merge_mode,
            "classes": [
                {"id": c.class_id, "name": c.name, "alias": c.alias, "definition": c.definition}
                for c in self.classes
            ],
            "scene_info": self.scene_info,
            "final_texts": final_texts,
            "confidence": confidence,
        }
        with open(os.path.join(self.output_dir, "semantic_metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        print(f"[Done] Saved semantic priors to: {self.output_dir}")
        print("- semantic_bank_combined.npy (recommended for current training)")


def parse_args():
    parser = argparse.ArgumentParser(description="Build hierarchical semantic priors for Chapter 4")
    parser.add_argument("--config", type=str, required=True, help="YAML config path")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    builder = SemanticPriorBuilder(cfg)
    builder.build()


if __name__ == "__main__":
    main()
