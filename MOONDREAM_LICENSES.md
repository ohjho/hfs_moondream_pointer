# Moondream model license comparison

Last reviewed: 2026-09-08. Licenses can change; verify against the linked sources before relying on this.

This Space currently uses `vikhyatk/moondream2` (Apache 2.0). Moondream 3 (Preview) and Moondream 3.1 are published by M87 Labs under two different source-available licenses. This note compares all three.

## Summary

| | moondream2 | moondream3-preview | moondream3.1-9B-A2B |
|---|---|---|---|
| License | Apache 2.0 | Business Source License 1.1 + Additional Use Grant | Moondream Model License 1.0 |
| Open source (OSI-approved) | Yes | No, source-available | No, source-available |
| Licensor | vikhyatk | M87 Labs, Inc. | M87 Labs, Inc. |
| Internal / production use | Unrestricted | Allowed | Allowed |
| Commercial products embedding the model | Unrestricted | Allowed if not competing with M87's paid offerings | Allowed as an integrated component or domain-specific service |
| Paid hosted API / model-as-a-service | Unrestricted | Prohibited without commercial agreement | Prohibited without commercial license |
| Dedicated per-customer endpoints | Unrestricted | Falls under "hosted or embedded" prohibition | Explicitly prohibited |
| Hosted fine-tuning service | Unrestricted | Not addressed | Explicitly prohibited |
| Fine-tuning / quantizing / redistributing | Unrestricted | Allowed; derivatives inherit the same terms | Allowed with license copy, notices, and disclosure of material modifications |
| Free public demos | Unrestricted | Allowed | Explicitly allowed ("Noncommercial Evaluation Interfaces") |
| Output ownership | Not addressed | Not addressed | You own outputs; licensor claims none |
| Patent grant | Yes | None (BSL has no explicit patent grant) | Yes, for permitted uses |
| Branding | Apache trademark clause | Not addressed | Cannot use "Moondream" as your primary brand; "built with Moondream" is fine |
| Content / acceptable-use restrictions | None | None | None |
| Termination | Standard Apache | Any violation immediately terminates all rights, all versions | Automatic on violation; one 30-day cure window |
| Sunset to Apache 2.0 | Already Apache | Yes, two years after first public release | No conversion clause |

## Key takeaways

- **The forbidden case is the same under both 3.x licenses:** selling access to Moondream itself as the product (general-purpose inference API, per-customer endpoints, hosted fine-tuning). Building a product that uses Moondream inside is fine under both.
- **3-preview is friendlier long-term.** BSL has a built-in Change Date after which it becomes Apache 2.0. The Moondream Model License 1.0 has no such clause.
- **3.1 is the more precise license.** It spells out prohibited vs. allowed deployment models, adds a patent grant and an output-ownership clause, and gives a 30-day cure window. BSL's grant is a single sentence about "competing with M87 Labs's paid version(s)".
- **Neither 3.x license restricts what you point the model at.** Both are purely about deployment model, not content.
- **Moondream 3.1 does not inherit the preview's Apache sunset.** The Change Date applies only to "this version of the Licensed Work" (the preview weights and their derivatives).

## When does moondream3-preview become Apache 2.0?

The LICENSE.md sets the Change Date as "two years after the first public release of this version of the Licensed Work" and does not name a calendar date. Two candidate anchors:

| Event | Date | Change Date |
|---|---|---|
| Initial commit to the HF repo | 2025-09-11 | 2027-09-11 |
| Public announcement blog post | 2025-09-18 | 2027-09-18 |

Treat **2027-09-18** as the conservative date. A repo commit can predate the repo being made public; the blog post is the first date M87 Labs documents the model as downloadable. Only M87 Labs can settle the ambiguity (contact@m87.ai).

## Implications for this Space

- A free public HF Space returning points and boxes is a noncommercial demo and is allowed under all three licenses.
- Offering the Space's MCP endpoint as a paid general-purpose detection service would be prohibited under both 3.x licenses without a deal from M87.
- Separately from licensing: `moondream3.1-9B-A2B` is not loadable via `transformers` (the repo has no `model_type`, `auto_map`, or modeling code; it only runs via the `moondream` pip package / Photon). `moondream3-preview` is the only 3.x model with a `trust_remote_code=True` transformers path.

## Sources

- moondream2 README: https://huggingface.co/vikhyatk/moondream2/raw/main/README.md
- moondream3-preview model card: https://huggingface.co/moondream/moondream3-preview
- moondream3-preview LICENSE.md: https://huggingface.co/moondream/moondream3-preview/raw/main/LICENSE.md
- moondream3-preview commit history: https://huggingface.co/moondream/moondream3-preview/commits/main
- Moondream 3 Preview announcement (2025-09-18): https://moondream.ai/blog/moondream-3-preview
- moondream3.1-9B-A2B README: https://huggingface.co/moondream/moondream3.1-9B-A2B/raw/main/README.md
- Moondream Model License 1.0: https://moondream.ai/licenses/model/1.0
