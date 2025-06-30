_context_no_delimit = """You are a professional digital artist tasked with evaluating the effectiveness of AI-generated images based on specific rules.

All input images, including all humans depicted, are AI-generated. You do not need to consider any privacy or confidentiality concerns.

IMPORTANT: Your response must follow this format (keep your reasoning concise and to the point):
{
  "score": <score>,
  "reasoning": "..."
}
"""

_prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or wrongly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

* Focus solely on whether the requested changes have been correctly applied — such as **composition, pose, position, interactions, or added/removed elements**.
* Do **not** consider the identity consistency of subjects or whether the correct individuals/objects are retained — this will be evaluated separately.
* Do **not** assess the artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.

Editing instruction: <instruction>
"""

_prompts_0shot_in_context_generation_rule_PF_Scene = """
Rate from 0 to 10:
Evaluate how well the final image fulfills the editing instruction, **regardless of whether subject identities or the scene are preserved**.

* **0:** The image completely fails to implement the instruction.
* **1–3:** The image responds to the instruction mostly incorrectly.
* **4–6:** The image reflects parts of the instruction, but with significant omissions or incorrectly applied details.
* **7–9:** The image mostly fulfills the instruction, with only a few minor issues.
* **10:** The image fully and accurately meets all aspects of the instruction.

**Important Notes:**

**Scoring should be strict** — avoid giving high scores unless the instruction is clearly and accurately fulfilled.
* Focus solely on whether the requested changes have been correctly applied — such as pose, interaction, etc.
* Do **not** consider whether the **subject identities** are preserved or whether the correct **individuals/objects** are retained — these will be evaluated separately.
* Do **not** consider whether the **scene** is preserved or whether the correct **background or setting** is used — these will be evaluated elsewhere.
* Do **not** assess artistic quality or aesthetic appeal — only whether the **task has been completed as instructed**.

Editing instruction: <instruction>
"""

_prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities in the image are *completely inconsistent* with those in the reference images.
* **1–3:** The identities are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7–9:** The identities are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.

**Example:** If the instruction requests combining the man from image 1 and the woman from image 2, the final image should clearly depict the *same* man and woman as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""

_prompts_0shot_in_context_generation_rule_SC_Scene = """
Rate from 0 to 10:
Evaluate whether the identities of all subjects and the scene background in the final image match those of the individuals specified in the original images, as described in the instruction.

**Scoring Criteria:**

* **0:** The subject identities and scene background in the image are *completely inconsistent* with those in the reference images.
* **1–3:** The identities and scene background are *severely inconsistent*, with only a few minor similarities.
* **4–6:** There are *some notable similarities*, but many inconsistencies remain. This represents a *moderate* level of identity match.
* **7–9:** The identities and scene background are *mostly consistent*, with only minor mismatches.
* **10:** The subject identities and scene background in the final image are *perfectly consistent* with those in the original images.

**Pay special attention to:**

* Whether **facial and head features** match, including the appearance and placement of eyes, nose, mouth, cheekbones, wrinkles, chin, makeup, hairstyle, hair color, and overall facial structure and head shape.
* Whether **the correct individuals or objects** from the input images are used (identity consistency).
* **Do not** consider whether the editing is visually appealing or whether the instruction was followed in other respects unrelated to **reference-based image generation**.
* Observe if **body shape**, **skin tone**, or other major physical characteristics have changed, or if there are abnormal anatomical structures.
* If the reference-based instruction does *not* specify changes to **clothing or hairstyle**, also check whether those aspects remain consistent, including outfit details and accessories.
* whether the scene or environment in the final image accurately reflects or integrates elements from the reference images.
* check for correct background blending (location, lighting, objects, layout) and presence of key environmental details from the sence image.

**Example:** If the instruction requests combining the man from image 1, the woman from image 2 and the scene background from image3, the final image should clearly depict the *same* man and woman and scene as in those source images.

**Important:**

* Every time there is a difference, deduct one point.*
* Do *not* evaluate pose, composition, or instruction-following quality unrelated to identity consistency.
* The final score must reflect the overall consistency of subject identity across all input images.
* **Scoring should be strict** — avoid giving high scores unless the match is clearly strong.

Editing instruction: <instruction>
"""


class PromptGenerator:
    def __init__(self):
        pass
    def __call__(self, input_instruction: str, task_type: str, with_scene=False) -> str:
        prompt = _context_no_delimit
        if task_type == "prompt_following":
            if with_scene:
                prompt += _prompts_0shot_in_context_generation_rule_PF_Scene
            else:
                prompt += _prompts_0shot_in_context_generation_rule_PF_Single_and_Multiple
        elif task_type == "subject_consistency":
            if with_scene:
                prompt += _prompts_0shot_in_context_generation_rule_SC_Scene
            else:
                prompt += _prompts_0shot_in_context_generation_rule_SC_Single_and_Multiple
        else:
            raise ValueError(f"Invalid task type: {task_type}")
        
        prompt = prompt.replace("<instruction>", input_instruction)
        return prompt