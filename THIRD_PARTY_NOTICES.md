# Third-party notices

The Apache 2.0 license in this repository applies to the project code. It does not grant rights to external datasets, model weights, or clinical records.

| Component | Source and terms |
|---|---|
| Shared training/model infrastructure | [SonoMed-VLM](https://github.com/jxia622/SonoMed-VLM), by Jack Xia, Apache 2.0. The `sonomed_vlm` package name is preserved for compatibility. |
| Qwen base model | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct), Qwen team. The model card lists Apache 2.0. No weights are redistributed. |
| Ultrasound adapter | Existing project checkpoint; not included in this release. Do not assume the code license licenses this checkpoint or its training corpus. |
| BreastVid | [CVA-Net author release](https://github.com/jhl-Det/CVA-Net#dataset-preparation). The authors permit non-commercial research and educational use. Dataset terms are distinct from the author's code license. No data are redistributed. Public correction identifiers implement the authors' [March 2023 errata](https://github.com/jhl-Det/CVA-Net#news). |
| ACOUSLIC-AI | [Version 1.1 training release](https://zenodo.org/records/12697994), DOI 10.5281/zenodo.12697994, CC BY-NC-SA 4.0. No data or annotation masks are redistributed. |
| Acquisition score reference | [Official ACOUSLIC evaluation implementation](https://github.com/DIAGNijmegen/ACOUSLIC-AI-evaluation-method/blob/c86c85c5e45440819b9d2ece32d0ca954a3b5256/src/acouslicaieval/evaluation.py). We use its per-frame quality convention in a different, best-of-acquired-set research endpoint. |

Python dependencies retain their own licenses. The inherited generic model loader also supports MedGemma; using those optional weights requires their separate model terms. This release's active-reading results use Qwen.
